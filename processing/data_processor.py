import os
import sys
from pathlib import Path

# Add project root to Python path to enable imports from utils
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import zarr
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from dotenv import load_dotenv
import pyproj

from utils.config import ConfigLoader
from utils.logging import LoggerSetup
from utils.data_manager import DataManager
from utils.data_checker import DataChecker



class SnowDataPipeline:
    """Integrated pipeline for checking, downloading, and processing snow data."""

    def __init__(self):
        """Initialize pipeline with configuration from environment."""
        # Setup environment
        self.env = os.getenv('DASHBOARD_ENV', 'local')

        # Determine config file paths based on project structure
        if self.env == 'aws':
            env_file = project_root / '.env'
            config_path = project_root / 'config' / 'config.aws.yaml'
        else:
            env_file = project_root / '.env'
            config_path = project_root / 'config' / 'config.local.yaml'

        # Load environment variables
        if env_file.exists():
            load_dotenv(env_file)

        # Initialize configuration
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config(self.env)

        # Setup logging
        self.logger_setup = LoggerSetup(self.config)
        self.logger = self.logger_setup.setup()

        # Initialize components
        self.data_manager = DataManager(self.config)
        self.data_checker = DataChecker(
            data_manager=self.data_manager,
            input_dir=Path(self.config['paths']['input_dir']),
            days_to_keep=self.config['retention_days']
        )

        # Create necessary directories
        Path(self.config['paths']['input_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)

        # Load and prepare mask
        mask_path = project_root / 'static' / 'OSMB-Taj-country-borders.geojson'
        self.mask_gdf = gpd.read_file(mask_path)
        self.bounds = self.mask_gdf.total_bounds

        # Initialize projections
        self.setup_projections()

    def setup_projections(self):
        """Setup projection transformers and bounds."""
        import pyproj

        # Get projection definitions from config
        input_proj = self.config['projections']['input']
        output_proj = self.config['projections']['output']

        # Create transformer
        self.transformer = pyproj.Transformer.from_crs(
            input_proj,
            output_proj,
            always_xy=True
        )

        # Get WGS84 bounds
        wgs84_bounds = self.config['projections']['bounds']['wgs84']

        # Transform bounds to Web Mercator
        min_x, min_y = self.transformer.transform(
            wgs84_bounds['min_lon'],
            wgs84_bounds['min_lat']
        )
        max_x, max_y = self.transformer.transform(
            wgs84_bounds['max_lon'],
            wgs84_bounds['max_lat']
        )

        # Store both sets of bounds
        self.bounds_wgs84 = {
            'min_lon': wgs84_bounds['min_lon'],
            'max_lon': wgs84_bounds['max_lon'],
            'min_lat': wgs84_bounds['min_lat'],
            'max_lat': wgs84_bounds['max_lat']
        }

        self.bounds_webmerc = {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y
        }

        self.logger.debug(f"WGS84 bounds: {self.bounds_wgs84}")
        self.logger.debug(f"Web Mercator bounds: {self.bounds_webmerc}")

    def create_mask_array(self, ds: xr.Dataset) -> np.ndarray:
        """Create a boolean mask array for the dataset based on Tajikistan bounds."""
        # Use WGS84 bounds for masking since input data is in WGS84
        lons = ds.lon.values
        lats = ds.lat.values
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)

        # Create mask for bounds using WGS84 coordinates
        mask = (
            (lon_mesh >= self.bounds_wgs84['min_lon']) &
            (lon_mesh <= self.bounds_wgs84['max_lon']) &
            (lat_mesh >= self.bounds_wgs84['min_lat']) &
            (lat_mesh <= self.bounds_wgs84['max_lat'])
        )

        # Expand mask to match data dimensions
        if 'time' in ds.sizes:
            mask = np.broadcast_to(
                mask[np.newaxis, :, :],
                (ds.sizes['time'], ds.sizes['lat'], ds.sizes['lon'])
            )

        return mask

    def _process_single_file(self, ds: xr.Dataset, var_name: str) -> xr.Dataset:
        """Process a single dataset."""
        try:
            # Create mask using WGS84 bounds
            mask = self.create_mask_array(ds)

            # Apply mask - replace values outside Tajikistan with NaN
            for var in ds.data_vars:
                if var != 'crs':  # Skip the CRS variable
                    ds[var] = ds[var].where(mask)

            # Clip to bounds using WGS84 coordinates
            ds = ds.sel(
                lon=slice(self.bounds_wgs84['min_lon'], self.bounds_wgs84['max_lon']),
                lat=slice(self.bounds_wgs84['min_lat'], self.bounds_wgs84['max_lat'])
            )

            # Transform coordinates to Web Mercator
            lons, lats = np.meshgrid(ds.lon.values, ds.lat.values)
            x_web, y_web = self.transformer.transform(lons, lats)

            # Add Web Mercator coordinates
            ds = ds.assign_coords({
                'x': ('lon', x_web[0, :]),
                'y': ('lat', y_web[:, 0])
            })

            # Add projection information
            ds.attrs['crs_wgs84'] = self.config['projections']['input']
            ds.attrs['crs_webmerc'] = self.config['projections']['output']
            ds.attrs['bounds_wgs84'] = str(self.bounds_wgs84)
            ds.attrs['bounds_webmerc'] = str(self.bounds_webmerc)

            # Add coordinate metadata
            ds['x'].attrs.update({
                'units': 'meters',
                'standard_name': 'projection_x_coordinate',
                'crs': self.config['projections']['output']
            })
            ds['y'].attrs.update({
                'units': 'meters',
                'standard_name': 'projection_y_coordinate',
                'crs': self.config['projections']['output']
            })

            return ds

        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            raise

    async def process_variable(self, var_name: str):
        """Process a single variable with proper time alignment."""
        self.logger.info(f"Processing variable: {var_name}")

        # Get dates to process
        today = datetime.now()
        dates = [today - timedelta(days=i) for i in range(self.config['retention_days'] + 1)]

        # Try to get latest file data
        latest_data = None
        reference_date = None

        for potential_date in dates[:2]:
            try:
                latest_data = await self.data_manager.get_data_for_date(var_name, potential_date)
                if latest_data is not None:
                    reference_date = potential_date
                    self.logger.info(f"Using reference date: {reference_date.strftime('%Y-%m-%d')} for {var_name}")
                    break
            except FileNotFoundError:
                self.logger.info(f"Data file not found for {var_name} on {potential_date.strftime('%Y-%m-%d')}, trying next date")
                continue
            except Exception as e:
                self.logger.error(f"Error accessing {var_name} on {potential_date.strftime('%Y-%m-%d')}: {e}")
                continue

        if latest_data is None:
            raise ValueError(f"No data available for {var_name} for today or yesterday")

        try:
            # Process latest file
            processed_data = self._process_single_file(latest_data, var_name)

            # Create proper time coordinates for forecast period
            forecast_times = [reference_date + timedelta(days=i) for i in range(5)]

            # Take first 5 time steps and assign proper time coordinates
            forecast_data = (processed_data
                            .isel(time=slice(0, 5))
                            .assign_coords(time=forecast_times))

            # Calculate accumulations with same time coordinates
            accumulated = (forecast_data.cumsum(dim='time')
                          .assign_coords(time=forecast_times))

            # Rename variables for forecast and accumulated
            forecast_data = forecast_data.rename({var_name: f"{var_name}_forecast"})
            accumulated = accumulated.rename({var_name: f"{var_name}_accumulated"})

            # Get historical data
            historical_data = []
            historical_times = []

            for date in dates[1:]:  # Skip reference date
                data = await self.data_manager.get_data_for_date(var_name, date)
                if data is not None:
                    # Take only the first time step
                    day1_data = self._process_single_file(data, var_name).isel(time=0)
                    historical_data.append(day1_data)
                    historical_times.append(date)

            # Create historical dataset if we have data
            if historical_data:
                historical_ds = xr.concat(historical_data, dim='time')
                historical_ds = historical_ds.assign_coords(time=historical_times)
                historical_ds = historical_ds.rename({var_name: f"{var_name}_historical"})

            # Log dimensions before merging
            self.logger.debug("Dataset dimensions before merging:")
            self.logger.debug(f"Forecast: time={forecast_data.sizes['time']}, values={forecast_times}")
            self.logger.debug(f"Accumulated: time={accumulated.sizes['time']}, values={forecast_times}")
            if historical_data:
                self.logger.debug(f"Historical: time={historical_ds.sizes['time']}, values={historical_times}")

            # Create the combined dataset
            combined_data = xr.merge([
                forecast_data,
                accumulated,
                historical_ds if historical_data else None
            ], join='outer')  # Use outer join to preserve all time points

            # Add metadata
            combined_data.attrs['reference_date'] = reference_date.strftime('%Y-%m-%d')
            combined_data.attrs['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            combined_data.attrs['forecast_start'] = forecast_times[0].strftime('%Y-%m-%d')
            combined_data.attrs['forecast_end'] = forecast_times[-1].strftime('%Y-%m-%d')
            if historical_data:
                combined_data.attrs['historical_start'] = historical_times[-1].strftime('%Y-%m-%d')
                combined_data.attrs['historical_end'] = historical_times[0].strftime('%Y-%m-%d')

            # Save processed data
            self._save_processed_data(combined_data, var_name)

        except Exception as e:
            self.logger.error(f"Error processing {var_name}: {e}")
            self.logger.exception("Detailed error:")
            raise
        finally:
            # Clean up
            latest_data.close()
            if 'data' in locals():
                data.close()

    def _save_processed_data(self, ds: xr.Dataset, var_name: str):
        """Save processed data in Zarr format with debugging information."""
        output_path = Path(self.config['paths']['output_dir']) / f"{var_name}_processed.zarr"

        # Log dataset information before saving
        self.logger.debug(f"Saving dataset with dimensions: {ds.sizes}")
        for var in ds.data_vars:
            self.logger.debug(f"Variable {var} shape: {ds[var].shape}")

        # Calculate chunk sizes based on actual dimensions
        chunk_sizes = {
            'time': min(1, ds.sizes['time']),
            'lat': min(100, ds.sizes['lat']),
            'lon': min(100, ds.sizes['lon']),
            'x': min(100, ds.sizes.get('x', 1)),
            'y': min(100, ds.sizes.get('y', 1))
        }

        # Set encoding for efficient storage and reading
        encoding = {}
        for var in ds.data_vars:
            if var == 'crs':
                # Special handling for crs variable which might be 1D
                encoding[var] = {
                    'chunks': chunk_sizes['time'],
                    'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
                }
            else:
                # For 3D variables
                encoding[var] = {
                    'chunks': (
                        chunk_sizes['time'],
                        chunk_sizes['lat'],
                        chunk_sizes['lon']
                    ),
                    'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
                }

        # Add encoding for coordinate variables
        for coord in ['x', 'y', 'lon', 'lat']:
            if coord in ds.coords:
                encoding[coord] = {
                    'chunks': chunk_sizes[coord],
                    'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
                }

        # Save to zarr format
        ds.to_zarr(output_path, mode='w', encoding=encoding)
        self.logger.info(f"Saved processed data to {output_path}")



async def main():
    """Main function to run the pipeline."""
    try:
        pipeline = SnowDataPipeline()
        for var_name in pipeline.data_manager.VARIABLES:
            await pipeline.process_variable(var_name)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())