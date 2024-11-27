# Data processing pipeline for snow data
#
# This script processes snow data files by transforming coordinates from
# EPSG:4326 to EPSG:3857, creating forecast and accumulated variables, and
# saving the processed data in Zarr format.
#
# The pipeline is designed to be run as an async function to allow for parallel
# processing of multiple variables.
#
# Useage:
# Requires pem file to access the snow data server. The path to the pem file
# should be set in the .env file as SSH_KEY_PATH. The root to SSH_KEY_PATH is
# the processing directory.
# To run locally, use the following command from the processing directory:
# python data_processor.py
#
# Author: Beatrice Marti, hydrosolutions GmbH

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

from scipy.ndimage import gaussian_filter  # For smoothing

import regionmask

from scipy import ndimage
from scipy.interpolate import griddata
import rasterio
from rasterio import features, transform

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
        mask_path = project_root / self.config['paths']['mask_path']
        self.mask_gdf = gpd.read_file(mask_path)
        self.bounds = self.mask_gdf.total_bounds


    def _process_single_file(self, ds: xr.Dataset, var_name: str) -> xr.Dataset:
        """
        Process a single dataset with consistent coordinate transformations:
        1. First slice to geojson mask
        2. Then create a mask using regionmask
        """
        self.logger.debug(f"=== _process_single_file ...")
        self.logger.debug(f"Processing single file {var_name}")
        self.logger.debug(f"Dataset shape before processing: {ds[var_name].shape}")

        # Get the resolution of ds. We want to buffer the slice by this amount
        res_lat = ds.lat.isel(lat=1).values - ds.lat.isel(lat=0).values
        res_lon = ds.lon.isel(lon=1).values - ds.lon.isel(lon=0).values
        self.logger.debug(f"Resolution of dataset: {res_lat}, {res_lon}")

        # Get the bounds for the slicing
        aoi_lat = [float(self.bounds[1]) - (res_lat*0.2),
                    float(self.bounds[3] + (res_lat*0.2))]
        aoi_lon = [float(self.bounds[0]) - (res_lon*0.2),
                    float(self.bounds[2] + (res_lon*0.2))]

        # Slice the dataset
        self.logger.debug(f"Slicing dataset to bounds: {aoi_lat}, {aoi_lon}")
        #ds_clip = ds.sel(lat=slice(aoi_lat[0], aoi_lat[1]),
        #                 lon=slice(aoi_lon[0], aoi_lon[1]))
        ds_clip = ds
        self.logger.debug(f"Dataset shape after slicing: {ds_clip[var_name].shape}")

        # Create a mask using regionmask
        self.logger.debug(f"Creating mask using regionmask")
        mask = regionmask.mask_3D_geopandas(
            self.mask_gdf,
            ds_clip.lon,
            ds_clip.lat)

        # Convert mask to binary (True/False) to avoid edge effects
        mask = mask == 1

        # Plot and save the mask
        if self.logger.level == logging.DEBUG:
            self.logger.debug(f"Plotting and saving the mask")
            mask.plot()
            # Save figure
            save_path = f"../data/processed/debugging/{var_name}_mask.png"
            # Expand the path to full path
            save_path = os.path.abspath(save_path)
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            except Exception as e:
                self.logger.error(f"Error creating directory: {str(e)}")

        # Apply the mask to the dataset
        self.logger.debug(f"Applying mask to dataset")
        ds_clip = ds_clip.where(mask)
        self.logger.debug(f"Dataset shape after masking: {ds_clip[var_name].shape}")

        # Smooth the data
        self.logger.debug(f"Smoothing the data")
        self.logger.debug(f"Dimensions of ds_clip: {ds_clip.dims}")

        # First, store the CRS information before dropping it
        crs_var = ds_clip['crs']

        # Drop the 'crs' variable for processing
        ds_clip = ds_clip.drop_vars('crs')

        # Define a smoothing function for use with apply_ufunc
        def smooth_data(array, sigma):
            """Apply Gaussian smoothing to a 2D array."""
            return gaussian_filter(array, sigma=sigma)

        # Apply the smoothing function to each time step
        smoothed_data = xr.apply_ufunc(
            smooth_data,
            ds_clip,  # The DataArray to process, only var_name values are passed
            input_core_dims=[["lat", "lon"]],  # Process over 2D slices (lat, lon)
            output_core_dims=[["lat", "lon"]],  # Output is also 2D
            vectorize=True,  # Apply the function independently to each time step
            kwargs={"sigma": 4},  # Pass the smoothing parameter
        )
        # Add crs attribute again
        # Add the CRS back as a variable to the smoothed data
        smoothed_data['crs'] = crs_var
        ds_clip = smoothed_data.copy()

        # If logger mode is set to debug, plot and save the masked data
        if self.logger.level == logging.DEBUG:
            self.logger.debug(f"Plotting and saving masked data")
            import matplotlib.pyplot as plt
            ds_clip[var_name].plot(col='time', col_wrap=4)
            #plt.show()
            # Save figure
            save_path = f"../data/processed/debugging/{var_name}_masked.png"
            # Expand the path to full path
            save_path = os.path.abspath(save_path)
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            except Exception as e:
                self.logger.error(f"Error creating directory: {str(e)}")
            try:
                plt.savefig(save_path)
                self.logger.debug(f"Saved masked data plot to {save_path}")
            except Exception as e:
                self.logger.error(f"Error saving masked data plot: {str(e)}")
            plt.close()

        # Transform to Web Mercator
        self.logger.debug(f"Transforming to Web Mercator")
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        lons, lats = np.meshgrid(ds_clip.lon.values, ds_clip.lat.values)
        x_web, y_web = transformer.transform(lons, lats)

        # Add Web Mercator coordinates
        ds_clip = ds_clip.assign_coords({
            "x": (("lon"), x_web[0, :]),
            "y": (("lat"), y_web[:, 0])
        })

        # print the dimensions of the dataset
        self.logger.debug(f"Dimensions of dataset after processing: {ds_clip.dims}")

        self.logger.debug(f"... _process_single_file ===")

        # Return the clipped dataset
        return ds_clip

    def generate_contours(self, combined_data: xr.Dataset, var_name: str) -> gpd.GeoDataFrame:
        """
        Generate filled contour polygons for both time series and accumulated data.
        Handles extra region dimension.
        """
        self.logger.debug(f"=== Generating contours for {var_name}")

        import matplotlib.pyplot as plt
        from shapely.geometry import Polygon
        import geopandas as gpd

        # Get predefined levels from config
        levels = self.config['variables'][var_name]['color_levels']
        self.logger.debug(f"Using predefined levels: {levels}")

        # Prepare storage for contour polygons
        contour_data = []

        # Variables to process
        var_suffixes = ['time_series', 'accumulated']

        for suffix in var_suffixes:
            full_var_name = f"{var_name}_{suffix}"
            self.logger.debug(f"\nProcessing {full_var_name}")

            # Check if variable exists in dataset
            if full_var_name not in combined_data:
                self.logger.warning(f"Variable {full_var_name} not found in dataset")
                continue

            # Get the variable
            data_var = combined_data[full_var_name]
            self.logger.debug(f"Variable dimensions: {data_var.dims}")
            self.logger.debug(f"Variable shape: {data_var.shape}")

            # Get coordinates
            x_coords = combined_data.lon.values
            y_coords = combined_data.lat.values
            X, Y = np.meshgrid(x_coords, y_coords)

            # Process each time step
            for t in range(len(combined_data.time)):
                self.logger.debug(f"\nProcessing time step {t}")
                try:
                    # Extract slice and squeeze out the region dimension
                    z = data_var.isel(time=t).squeeze('region').values

                    self.logger.debug(f"Processed slice shape: {z.shape}")
                    self.logger.debug(f"Min value: {np.nanmin(z)}, Max value: {np.nanmax(z)}")

                    current_time = combined_data.time[t].values

                    if np.all(np.isnan(z)):
                        self.logger.warning(f"Skipping time step {t} - all values are NaN")
                        continue

                    # Generate filled contours
                    fig, ax = plt.subplots()
                    try:
                        contours = ax.contourf(X, Y, z, levels=levels, extend='both')

                        for i, collection in enumerate(contours.collections):
                            if i == 0:
                                level_value = levels[0] / 2
                            elif i == len(levels):
                                level_value = levels[-1] * 1.5
                            else:
                                level_value = (levels[i-1] + levels[i]) / 2

                            for path in collection.get_paths():
                                if len(path.vertices) > 2:
                                    try:
                                        poly = Polygon(path.vertices).buffer(0)
                                        if poly.is_valid and not poly.is_empty:
                                            contour_data.append({
                                                'geometry': poly,
                                                'level': level_value,
                                                'time': current_time,
                                                'level_idx': i,
                                                'variable': full_var_name
                                            })
                                    except Exception as e:
                                        self.logger.warning(f"Failed to create polygon: {e}")

                    except Exception as e:
                        self.logger.error(f"Error in contour generation: {str(e)}")
                        self.logger.debug(f"Z array info - shape: {z.shape}, dtype: {z.dtype}")
                        self.logger.debug(f"X shape: {X.shape}, Y shape: {Y.shape}")
                    finally:
                        plt.close(fig)

                except Exception as e:
                    self.logger.error(f"Error processing time step {t}: {str(e)}")
                    continue

        if not contour_data:
            self.logger.error("No valid contour data generated")
            return None

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(contour_data, crs="EPSG:3857")

        # Add metadata as attributes
        gdf.attrs['levels'] = levels
        gdf.attrs['variables'] = [f"{var_name}_{suffix}" for suffix in var_suffixes]

        self.logger.debug(f"Created contour dataset with {len(contour_data)} polygons")
        self.logger.debug("=== Finished generating contours")

        return gdf

    def check_datasets(self, historical_ds, forecast_data):
        """Check datasets for compatibility before concatenation"""
        # Print shapes
        print(f"Historical shape: {historical_ds.dims}")
        print(f"Forecast shape: {forecast_data.dims}")

        # Print variables
        print(f"Historical variables: {list(historical_ds.data_vars)}")
        print(f"Forecast variables: {list(forecast_data.data_vars)}")

        # Print time ranges
        print(f"Historical time range: {historical_ds.time.values[0]} to {historical_ds.time.values[-1]}")
        print(f"Forecast time range: {forecast_data.time.values[0]} to {forecast_data.time.values[-1]}")

        return all(
            var in forecast_data.data_vars
            for var in historical_ds.data_vars
        )

    def concat_time_series(self, historical_ds, forecast_data):
        """Safely concatenate historical and forecast datasets"""
        try:
            # Check compatibility
            # if logger mode is debug only
            if self.logger.level == logging.DEBUG:
                if not self.check_datasets(historical_ds, forecast_data):
                    raise ValueError("Datasets have incompatible variables")

            # Concatenate along time dimension
            time_series = xr.concat(
                [historical_ds, forecast_data],
                dim='time',
                combine_attrs='override'  # Use this if attributes differ
            )

            # Sort by time to ensure chronological order
            time_series = time_series.sortby('time')

            # Print result info
            print(f"Combined time range: {time_series.time.values[0]} to {time_series.time.values[-1]}")
            print(f"Total timesteps: {len(time_series.time)}")

            return time_series

        except Exception as e:
            print(f"Error concatenating datasets: {str(e)}")
            raise

    async def process_variable(self, var_name: str):
        """Process a single variable with proper time alignment."""
        self.logger.debug(f"=== process_variable ...")
        self.logger.info(f"Processing variable: {var_name}")

        # Get dates to process
        today = datetime.now()
        dates = [today - timedelta(days=i) for i in range(self.config['retention_days'] + 1)]

        # Try to get latest file data
        latest_data = None
        reference_date = None

        self.logger.debug(f"Get data for dates: {dates}")
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

        self.logger.debug(f"Latest data shape: {latest_data[var_name].shape}")

        try:
            # Process latest file
            processed_data = self._process_single_file(latest_data, var_name)
            self.logger.debug(f"Processed data shape: {processed_data[var_name].shape}")

            # Create proper time coordinates for forecast period
            forecast_times = [reference_date + timedelta(days=i) for i in range(5)]
            forecast_times_set = set(forecast_times)  # Convert to set for efficient lookup

            # Take first 5 time steps and assign proper time coordinates
            forecast_data = (processed_data
                            .isel(time=slice(0, 5))
                            .assign_coords(time=forecast_times))
            self.logger.debug(f"Forecast data shape: {forecast_data[var_name].shape}")

            # Calculate accumulations with same time coordinates
            accumulated = (forecast_data.cumsum(dim='time')
                          .assign_coords(time=forecast_times))

            # Subtract the first time step to get the actual values
            accumulated = accumulated - accumulated.isel(time=0)

            # Rename variables for forecast and accumulated
            forecast_data = forecast_data.rename({var_name: f"{var_name}_time_series"})
            accumulated = accumulated.rename({var_name: f"{var_name}_accumulated"})

            # Get historical data
            historical_data = []
            historical_times = []

            # We process historical data only from the start of dates up to one
            # day before the reference date.

            for date in dates[1:]:  # Skip reference date
                # Skip if date is in forecast period
                if date in forecast_times_set:
                    continue

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
                historical_ds = historical_ds.rename({var_name: f"{var_name}_time_series"})

            # Log dimensions before merging
            self.logger.debug("Dataset dimensions before merging:")
            self.logger.debug(f"Forecast: time={forecast_data.sizes['time']}, values={forecast_times}")
            self.logger.debug(f"Accumulated: time={accumulated.sizes['time']}, values={forecast_times}")
            if historical_data:
                self.logger.debug(f"Historical: time={historical_ds.sizes['time']}, values={historical_times}")

            # Get historical_ds, if it exists, in one dataset together with forecast_data
            self.logger.debug("Merging historical and forecast datasets...")
            self.logger.debug(f"Times in forecast_data: {forecast_data.time.values}")
            if historical_data:
                self.logger.debug(f"Times in historical_ds: {historical_ds.time.values}")
                time_series = self.concat_time_series(historical_ds, forecast_data)
            else:
                time_series = forecast_data

           # Sort by time to ensure chronological order
            time_series = time_series.sortby('time')

            # Create the combined dataset
            combined_data = xr.merge([
                time_series,
                accumulated
            ], join='outer')  # Use outer join to preserve all time points

            # Add metadata
            combined_data.attrs['reference_date'] = reference_date.strftime('%Y-%m-%d')
            combined_data.attrs['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            combined_data.attrs['forecast_start'] = forecast_times[0].strftime('%Y-%m-%d')
            combined_data.attrs['forecast_end'] = forecast_times[-1].strftime('%Y-%m-%d')
            if historical_data:
                combined_data.attrs['historical_start'] = historical_times[-1].strftime('%Y-%m-%d')
                combined_data.attrs['historical_end'] = historical_times[0].strftime('%Y-%m-%d')

            '''
            # Generate contours
            contour_gdf = self.generate_contours(combined_data, var_name)
            if contour_gdf is not None:
                # Save contour data
                contour_file = f"{self.config['paths']['output_dir']}/{var_name}_contours.gpkg"
                contour_gdf.to_file(contour_file, driver='GPKG')
                if self.logger.level == logging.DEBUG:
                    self.logger.debug(f"Saved contour data to {contour_file}")
                    contour_path = Path(self.config['paths']['output_dir']) / f"{var_name}_contours.geojson"
                    contour_gdf.to_file(contour_path, driver='GeoJSON')
                self.logger.info(f"Saved contour data to {contour_path}")
            '''

            # Save raster data
            self._save_processed_data(combined_data, var_name)

        except Exception as e:
            self.logger.error(f"Error processing {var_name}: {e}")
            self.logger.exception("Detailed error:")
            raise
        finally:
            # Clean up
            latest_data.close()
            if 'data' in locals():
                # Close data if it is not None
                if data is not None:
                    data.close()

    def _save_processed_data(self, ds: xr.Dataset, var_name: str):
        """Save processed data in Zarr format with debugging information."""
        self.logger.debug(f"=== _save_processed_data ...")
        self.logger.debug(f"Saving processed data for {var_name}")
        output_path = Path(self.config['paths']['output_dir']) / f"{var_name}_processed.zarr"

        # Create a copy of the dataset
        ds_fixed = ds.copy()

        # Set CRS variable
        ds_fixed['crs'] = xr.DataArray(
            data=0,  # Placeholder value
            attrs={
                'spatial_ref': 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]',
                'grid_mapping_name': 'mercator',
                'epsg_code': 'EPSG:3857'
            }
        )

        # Update coordinate attributes
        ds_fixed['x'].attrs.update({
            'units': 'meters',
            'standard_name': 'projection_x_coordinate',
            'axis': 'X'
        })
        ds_fixed['y'].attrs.update({
            'units': 'meters',
            'standard_name': 'projection_y_coordinate',
            'axis': 'Y'
        })

        # Set proper encoding
        encoding = {}
        compressor = zarr.Blosc(cname='lz4', clevel=5, shuffle=1)

        # Encode CRS (scalar variable)
        encoding['crs'] = {
            'chunks': None,  # Scalar variable
            'compressor': compressor
        }

        # Encode main data variables
        for var in ['hs_time_series', 'hs_accumulated']:
            if var in ds_fixed.data_vars:
                encoding[var] = {
                    'chunks': (1, 500, 500),  # time, lat, lon
                    'compressor': compressor
                }

        # Encode coordinates
        encoding.update({
            'x': {'chunks': -1, 'compressor': compressor},  # Store as single chunk
            'y': {'chunks': -1, 'compressor': compressor},  # Store as single chunk
            'lon': {'chunks': -1, 'compressor': compressor},
            'lat': {'chunks': -1, 'compressor': compressor},
            'time': {'chunks': -1, 'compressor': compressor},
            'region': {'chunks': -1, 'compressor': compressor}
        })

        # Save to zarr format
        self.logger.debug(f"Starting zarr write operation...")
        ds_fixed.to_zarr(output_path, mode='w', encoding=encoding, consolidated=True)
        self.logger.info(f"Saved processed data to {output_path}")
        self.logger.debug(f"... _save_processed_data ===")

    def delete_old_files(self):
        """Remove all files older than the retention period."""
        try:
            # Get all files in the output directory
            output_dir = Path(self.config['paths']['cache_dir'])
            all_files = list(output_dir.glob('*.nc'))

            # Get the retention period
            retention_days = self.config['retention_days']

            # Get the date to compare against
            today = datetime.now()

            # Filter files older than the retention period
            old_files = [
                f for f in all_files
                if (today - datetime.fromtimestamp(f.stat().st_mtime)).days > retention_days
            ]
            # Delete old files
            for f in old_files:
                f.unlink()
                self.logger.info(f"Deleted old file: {f.name}")

            self.logger.info(f"Deleted {len(old_files)} files older than {retention_days} days")

        except Exception as e:
            self.logger.error(f"Error deleting old files: {e}")
            raise

async def main():
    """Main function to run the pipeline."""
    try:
        pipeline = SnowDataPipeline()
        for var_name in pipeline.data_manager.VARIABLES:
            await pipeline.process_variable(var_name)
        pipeline.delete_old_files()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())