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

    '''
    def _process_single_file(self, ds: xr.Dataset, var_name: str) -> xr.Dataset:
        """
        Process a single dataset by transforming coordinates from EPSG:4326 to EPSG:3857.

        Args:
            ds (xr.Dataset): Input dataset containing the variable to process (in EPSG:4326)
            var_name (str): Name of the variable being processed

        Returns:
            xr.Dataset: Dataset with transformed coordinates (in EPSG:3857)
        """
        try:
            # Create a copy of the dataset to avoid modifying the original
            processed_ds = ds.copy()

            # Create transformer for EPSG:4326 (WGS 84) to EPSG:3857 (Web Mercator)
            transformer = pyproj.Transformer.from_crs(
                "EPSG:4326",
                "EPSG:3857",
                always_xy=True
            )

            # Get original coordinates
            lons, lats = np.meshgrid(ds.lon.values, ds.lat.values)

            # Transform coordinates
            x_web, y_web = transformer.transform(lons, lats)

            # Create new coordinates
            processed_ds = processed_ds.assign_coords({
                "x": (("lon"), x_web[0, :]),  # Use first row for x coordinates
                "y": (("lat"), y_web[:, 0])   # Use first column for y coordinates
            })

            # Add metadata about the transformation
            processed_ds.attrs.update({
                'original_crs': 'EPSG:4326',
                'transformed_crs': 'EPSG:3857',
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            # Log the transformation
            self.logger.info(f"Transformed coordinates for {var_name} from EPSG:4326 to EPSG:3857")
            self.logger.debug(f"X range: {x_web.min():.2f} to {x_web.max():.2f}")
            self.logger.debug(f"Y range: {y_web.min():.2f} to {y_web.max():.2f}")

            return processed_ds

        except Exception as e:
            self.logger.error(f"Error in _process_single_file: {str(e)}")
            raise
    '''

    '''def _process_single_file(self, ds: xr.Dataset, var_name: str) -> xr.Dataset:
        """
        Process a single dataset by:
        1. Transforming coordinates from EPSG:4326 to EPSG:3857
        2. Clipping to Tajikistan boundary from GeoJSON

        Args:
            ds (xr.Dataset): Input dataset containing the variable to process (in EPSG:4326)
            var_name (str): Name of the variable being processed

        Returns:
            xr.Dataset: Dataset with transformed coordinates and clipped to boundary
        """
        try:
            # Create a copy of the dataset to avoid modifying the original
            processed_ds = ds.copy()

            # Create transformer for EPSG:4326 (WGS 84) to EPSG:3857 (Web Mercator)
            transformer = pyproj.Transformer.from_crs(
                "EPSG:4326",
                "EPSG:3857",
                always_xy=True
            )

            # Get original coordinates
            lons, lats = np.meshgrid(ds.lon.values, ds.lat.values)

            # Transform coordinates
            x_web, y_web = transformer.transform(lons, lats)

            # Create new coordinates
            processed_ds = processed_ds.assign_coords({
                "x": (("lon"), x_web[0, :]),  # Use first row for x coordinates
                "y": (("lat"), y_web[:, 0])   # Use first column for y coordinates
            })

            # Calculate the resolution of the data
            lon_res = np.abs(ds.lon.values[1] - ds.lon.values[0])
            lat_res = np.abs(ds.lat.values[1] - ds.lat.values[0])

            # Create transform for the rasterization
            from affine import Affine
            transform = Affine.translation(lons.min(), lats.min()) * Affine.scale(lon_res, lat_res)

            # Create mask using rasterio's geometry mask
            from rasterio.features import geometry_mask

            # Ensure mask_gdf is in EPSG:4326 (same as input data)
            mask_gdf = self.mask_gdf.to_crs("EPSG:4326")

            # Create the mask
            mask = ~geometry_mask(
                mask_gdf.geometry,
                out_shape=(len(ds.lat), len(ds.lon)),
                transform=transform,
                invert=True
            )

            # Expand mask if we have a time dimension
            if 'time' in ds.dims:
                mask = np.broadcast_to(
                    mask[np.newaxis, :, :],
                    (ds.sizes['time'], ds.sizes['lat'], ds.sizes['lon'])
                )

            # Apply the mask to each data variable
            for var in processed_ds.data_vars:
                if var != 'crs':  # Skip CRS variable
                    processed_ds[var] = processed_ds[var].where(mask)

            # Get bounds of the mask for clipping
            bounds = mask_gdf.total_bounds  # [minx, miny, maxx, maxy]
            buffer = max(lon_res, lat_res)  # Use one grid cell as buffer

            # Clip to the bounds of the polygon with buffer
            processed_ds = processed_ds.sel(
                lon=slice(bounds[0] - buffer, bounds[2] + buffer),
                lat=slice(bounds[3] + buffer, bounds[1] - buffer)  # Note reversed order for latitude
            )

            # Verify we have data after clipping
            if processed_ds.sizes['lat'] == 0 or processed_ds.sizes['lon'] == 0:
                self.logger.error(f"No data found within bounds for {var_name}")
                return None

            # Add metadata about the processing
            processed_ds.attrs.update({
                'original_crs': 'EPSG:4326',
                'transformed_crs': 'EPSG:3857',
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'spatial_coverage': 'Tajikistan',
                'clipping_bounds': str(bounds),
                'spatial_resolution': f'{lon_res} degrees'
            })

            # Log the transformation and clipping
            self.logger.info(f"Processed {var_name}: transformed coordinates and clipped to boundary")
            self.logger.debug(f"Original shape: {ds.sizes}")
            self.logger.debug(f"Processed shape: {processed_ds.sizes}")

            return processed_ds

        except Exception as e:
            self.logger.error(f"Error in _process_single_file: {str(e)}")
            raise'''


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
        aoi_lat = [float(self.bounds[1]) - res_lat,
                    float(self.bounds[3] + res_lat)]
        aoi_lon = [float(self.bounds[0]) - res_lon,
                    float(self.bounds[2] + res_lon)]

        # Slice the dataset
        self.logger.debug(f"Slicing dataset to bounds: {aoi_lat}, {aoi_lon}")
        ds_clip = ds.sel(lat=slice(aoi_lat[0], aoi_lat[1]),
                         lon=slice(aoi_lon[0], aoi_lon[1]))
        self.logger.debug(f"Dataset shape after slicing: {ds_clip[var_name].shape}")

        # Create a mask using regionmask
        self.logger.debug(f"Creating mask using regionmask")
        mask = regionmask.mask_3D_geopandas(self.mask_gdf, ds_clip.lon, ds_clip.lat)

        # Apply the mask to the dataset
        self.logger.debug(f"Applying mask to dataset")
        ds_clip = ds_clip.where(mask)
        self.logger.debug(f"Dataset shape after masking: {ds_clip[var_name].shape}")

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


    '''def _process_single_file(self, ds: xr.Dataset, var_name: str) -> xr.Dataset:
        """
        Process a single dataset with consistent coordinate transformations:
        1. First clip using rasterio's mask in EPSG:4326
        2. Then transform to Web Mercator (EPSG:3857)
        3. Finally apply smoothing and interpolation
        """
        try:
            # Create a copy of the dataset to avoid modifying the original
            self.logger.debug("\n=== Starting Diagnostic Process ===")
            self.logger.debug("Creating copy of dataset...")# Create a copy of the dataset
            processed_ds = ds.copy()

            # Print original data information
            self.logger.debug("\nOriginal Dataset Information:")
            self.logger.debug(f"Dimensions: {ds.dims}")
            self.logger.debug(f"Coordinates:\nLat: {ds.lat.values[:5]} ... {ds.lat.values[-5:]}")
            self.logger.debug(f"Lon: {ds.lon.values[:5]} ... {ds.lon.values[-5:]}")

            # Plot original data
            if self.logger.level == logging.DEBUG:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 8))
                plt.subplot(121)
                ds[var_name][0].plot()
                plt.title("Original Data")
                plt.savefig(f"../data/processed/debugging/{var_name}_original.png")
                plt.close()

            # Print original bounding box and crs
            print(f"---------------------------------")
            self.logger.debug(f"Processing {var_name}")

            # Step 1: Mask in WGS84 (EPSG:4326) using rasterio's fast mask
            # --------------------------------------------
            mask_gdf_4326 = self.mask_gdf.to_crs("EPSG:4326")
            self.logger.debug(f"Mask bounds: {mask_gdf_4326.total_bounds}")
            shapes = [feature['geometry'] for feature in mask_gdf_4326.geometry.__geo_interface__['features']]
            self.logger.debug(f"Number of shapes: {len(shapes)}")

            def mask_with_rasterio(data_array, shapes, nodata=np.nan):
                """Mask a 2D array using rasterio's fast mask operation"""
                self.logger.debug("\n--- Masking Single Array ---")
                # Convert xarray to numpy array
                data = data_array.values

                # Get the coordinates in the correct order
                height, width = data.shape
                self.logger.debug(f"Array shape: {height} x {width}")

                left = float(data_array.lon.min())
                right = float(data_array.lon.max())
                bottom = float(data_array.lat.min())
                top = float(data_array.lat.max())

                self.logger.debug(f"Coordinate bounds:")
                self.logger.debug(f"Longitude: {left} to {right}")
                self.logger.debug(f"Latitude: {bottom} to {top}")

                # Create transform with north-up orientation
                transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
                self.logger.debug(f"Transform matrix: {transform}")

                # Plot data before masking
                if self.logger.level == logging.DEBUG:
                    plt.figure(figsize=(12, 8))
                    plt.subplot(121)
                    plt.imshow(data)
                    plt.title("Data Before Masking")
                    plt.subplot(122)
                    plt.imshow(np.flipud(data))
                    plt.title("Data Flipped")
                    plt.savefig(f"../data/processed/debugging/{var_name}_before_mask.png")
                    plt.close()

                # Create a temporary rasterio dataset in memory
                with rasterio.io.MemoryFile() as memfile:
                    with memfile.open(
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=1,
                        dtype=data.dtype,
                        crs='EPSG:4326',
                        transform=transform,
                        nodata=nodata
                    ) as dataset:
                        # Write the data with correct orientation
                        #dataset.write(np.flipud(data).reshape(1, height, width))
                        flipped_data = np.flipud(data)
                        self.logger.debug(f"Writing data to temporary dataset...")
                        dataset.write(flipped_data.reshape(1, height, width))

                        # Perform the masking
                        masked_data, _ = rasterio.mask.mask(
                            dataset,
                            shapes,
                            nodata=nodata,
                            crop=False,
                            filled=True
                        )

                # Flip the result back to match the original orientation
                #result = np.flipud(masked_data[0])
                result = masked_data[0]

                # Plot masked result
                if self.logger.level == logging.DEBUG:
                    plt.figure(figsize=(12, 8))
                    plt.subplot(121)
                    plt.imshow(masked_data[0])
                    plt.title("Direct Mask Result")
                    plt.subplot(122)
                    plt.imshow(result)
                    plt.title("Final Result (Flipped)")
                    plt.savefig(f"../data/processed/debugging/{var_name}_mask_result.png")
                    plt.close()

                return result

            # Apply masking to each time step
            self.logger.debug("\n=== Applying Masking ===")
            if 'time' in processed_ds.dims:
                masked_arrays = []
                for time_idx in range(len(processed_ds.time)):
                    self.logger.debug(f"Processing time step {time_idx + 1}/{len(processed_ds.time)}")
                    data_slice = processed_ds[var_name].isel(time=time_idx)
                    masked_arrays.append(mask_with_rasterio(data_slice, shapes))
                masked_data = np.stack(masked_arrays)
                processed_ds[var_name] = (('time', 'lat', 'lon'), masked_data)
            else:
                masked_data = mask_with_rasterio(processed_ds[var_name], shapes)
                processed_ds[var_name] = (('lat', 'lon'), masked_data)

            # Log and plot final result
            self.logger.debug("\n=== Final Masking Result ===")
            self.logger.debug(f"Final array shape: {processed_ds[var_name].shape}")

            if self.logger.level == logging.DEBUG:
                plt.figure(figsize=(12, 8))
                if 'time' in processed_ds.dims:
                    plt.imshow(processed_ds[var_name][0])
                else:
                    plt.imshow(processed_ds[var_name])
                plt.title("Final Processed Result")
                plt.colorbar()
                plt.savefig(f"../data/processed/debugging/{var_name}_final.png")
                plt.close()

            # Step 2: Transform to Web Mercator
            # -------------------------------------------------
            self.logger.debug("\n=== Step 2: Transforming to Web Mercator ===")
            transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            lons, lats = np.meshgrid(processed_ds.lon.values, processed_ds.lat.values)
            x_web, y_web = transformer.transform(lons, lats)

            # Add Web Mercator coordinates
            processed_ds = processed_ds.assign_coords({
                "x": (("lon"), x_web[0, :]),
                "y": (("lat"), y_web[:, 0])
            })
            self.logger.debug(f"Transformed coordinates for {var_name} from EPSG:4326 to EPSG:3857")
            self.logger.debug(f"Type of processed_ds: {type(processed_ds)}")
            self.logger.debug(f"Dimensions of processed_ds: {processed_ds.dims}")

            # if logger mode is debug, plot and save the transformed data
            if self.logger.level == logging.DEBUG:
                import matplotlib.pyplot as plt
                plt.imshow(processed_ds[var_name][0])
                plt.colorbar()
                plt.savefig(f"../data/processed/debugging/{var_name}_transformed.png")
                plt.close()

            # Step 3: Apply smoothing and interpolation (if enabled)
            # --------------------------------------------------
            self.logger.debug("\n=== Step 3: Smoothing and interpolation ===")
            if self.config.get('visualization', {}).get('enable_optimization', True):
                smooth_factor = self.config.get('visualization', {}).get('smoothing_factor', 1.0)
                upscale_factor = self.config.get('visualization', {}).get('upscale_factor', 2.0)

                new_x = np.linspace(x_web.min(), x_web.max(),
                                  int(len(processed_ds.lon) * upscale_factor))
                new_y = np.linspace(y_web.min(), y_web.max(),
                                  int(len(processed_ds.lat) * upscale_factor))

                if 'time' in processed_ds.dims:
                    smoothed_data = []
                    for t in range(len(processed_ds.time)):
                        slice_data = processed_ds[var_name].isel(time=t)
                        smoothed = self._smooth_and_interpolate(
                            slice_data.values,
                            x_web[0, :],
                            y_web[:, 0],
                            new_x,
                            new_y,
                            smooth_factor
                        )
                        smoothed_data.append(smoothed)

                    data_array = xr.DataArray(
                        np.stack(smoothed_data),
                        dims=['time', 'y', 'x'],
                        coords={
                            'time': processed_ds.time,
                            'y': new_y,
                            'x': new_x
                        }
                    )
                else:
                    smoothed = self._smooth_and_interpolate(
                        processed_ds[var_name].values,
                        x_web[0, :],
                        y_web[:, 0],
                        new_x,
                        new_y,
                        smooth_factor
                    )
                    data_array = xr.DataArray(
                        smoothed,
                        dims=['y', 'x'],
                        coords={'y': new_y, 'x': new_x}
                    )

                processed_ds[var_name] = data_array

            # if logger mode is debug, plot and save the smoothed data
            if self.logger.level == logging.DEBUG:
                import matplotlib.pyplot as plt
                plt.imshow(data_array.values)
                plt.colorbar()
                plt.savefig(f"../data/processed/debugging/{var_name}_smoothed.png")
                plt.close()

            # Add metadata
            processed_ds.attrs.update({
                'original_crs': 'EPSG:4326',
                'transformed_crs': 'EPSG:3857',
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'spatial_coverage': 'Tajikistan',
                'web_mercator_resolution': f'{np.mean(np.diff(x_web[0, :])):.2f} meters',
                'visualization_optimization': str(self.config.get('visualization', {}).get('enable_optimization', True)),
                'smoothing_factor': self.config.get('visualization', {}).get('smoothing_factor', 1.0),
                'upscale_factor': self.config.get('visualization', {}).get('upscale_factor', 2.0)
            })

            return processed_ds

        except Exception as e:
            self.logger.error(f"Error in _process_single_file: {str(e)}")
            raise'''

    def _smooth_and_interpolate(
        self,
        data: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        new_x: np.ndarray,
        new_y: np.ndarray,
        smooth_factor: float
    ) -> np.ndarray:
        """
        Helper method to smooth and interpolate data for better visualization.
        """
        # Create meshgrids for interpolation
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        new_x_mesh, new_y_mesh = np.meshgrid(new_x, new_y)

        # Remove NaN values for interpolation
        valid_mask = ~np.isnan(data)
        points = np.column_stack((x_mesh[valid_mask].ravel(), y_mesh[valid_mask].ravel()))
        values = data[valid_mask].ravel()

        # Interpolate to higher resolution grid
        interpolated = griddata(
            points,
            values,
            (new_x_mesh, new_y_mesh),
            method='cubic',
            fill_value=np.nan
        )

        # Apply smoothing if factor > 0
        if smooth_factor > 0:
            interpolated = ndimage.gaussian_filter(
                interpolated,
                sigma=smooth_factor,
                mode='nearest'
            )

        return interpolated

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
                # Close data if it is not None
                if data is not None:
                    data.close()

    '''def _save_processed_data(self, ds: xr.Dataset, var_name: str):
        """Save processed data in Zarr format with debugging information."""
        self.logger.debug(f"=== _save_processed_data ...")
        self.logger.debug(f"Saving processed data for {var_name}")
        output_path = Path(self.config['paths']['output_dir']) / f"{var_name}_processed.zarr"

        # Log dataset information before saving
        self.logger.debug(f"Saving dataset with dimensions: {ds.sizes}")
        # Log ds type and variables
        self.logger.debug(f"Dataset type: {type(ds)}")
        self.logger.debug(f"Dataset variables: {list(ds.data_vars)}")
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

        self.logger.debug(f"Chunk sizes: {chunk_sizes}")

        # Set encoding for efficient storage and reading
        encoding = {}
        for var in ds.data_vars:
            if var == 'region' or var == 'crs':
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
        self.logger.debug(f"Encoding: {encoding}")

        # Add encoding for coordinate variables
        self.logger.debug(f"Coodinates in dataset: {ds.coords}")
        for coord in ['x', 'y', 'lon', 'lat']:
            if coord in ds.coords:
                encoding[coord] = {
                    'chunks': chunk_sizes[coord],
                    'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
                }
        self.logger.debug(f"Encoding for coordinates: {encoding}")

        # Save to zarr format
        ds.to_zarr(output_path, mode='w', encoding=encoding)
        self.logger.info(f"Saved processed data to {output_path}")

        self.logger.debug(f"... _save_processed_data ===")'''
    '''
    def _save_processed_data(self, ds: xr.Dataset, var_name: str):
        """Save processed data in Zarr format with debugging information."""
        self.logger.debug(f"=== _save_processed_data ...")
        self.logger.debug(f"Saving processed data for {var_name}")
        output_path = Path(self.config['paths']['output_dir']) / f"{var_name}_processed.zarr"

        # Debug data types before conversion
        self.logger.debug("Dataset dtypes before conversion:")
        for var in ds.variables:
            self.logger.debug(f"{var}: {ds[var].dtype}")

        # Create a copy of the dataset
        ds_fixed = ds.copy()

        # Ensure proper CRS handling
        import rioxarray  # Make sure this is imported at the top

        # Set CRS if not already set
        if not hasattr(ds_fixed, 'rio') or ds_fixed.rio.crs is None:
            self.logger.debug("Setting CRS to EPSG:3857 (Web Mercator)")
            ds_fixed.rio.write_crs("EPSG:3857", inplace=True)  # Web Mercator

        ds_fixed.attrs.update({
            'crs': "EPSG:3857",
            'spatial_ref': 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]'
        })
        # Special handling for CRS
        #if 'crs' in ds_fixed:
        #    # Convert CRS to string attributes instead of a variable
        #    crs_attrs = ds_fixed.crs.attrs
        #    ds_fixed = ds_fixed.drop_vars('crs')
        #    # Add CRS attributes to the dataset
        #    ds_fixed.attrs.update({
        #        'crs': str(crs_attrs.get('spatial_ref', '')),
        #        'grid_mapping': 'crs'
        #    })

        # Calculate chunk sizes
        chunk_sizes = {
            'time': 1,
            'lat': min(500, ds_fixed.sizes['lat']),
            'lon': min(500, ds_fixed.sizes['lon']),
            'x': min(500, ds_fixed.sizes.get('x', 1)),
            'y': min(500, ds_fixed.sizes.get('y', 1))
        }

        # Pre-chunk the dataset
        ds_fixed = ds_fixed.chunk({
            'time': chunk_sizes['time'],
            'lat': chunk_sizes['lat'],
            'lon': chunk_sizes['lon']
        })

        # Set encoding
        compressor = zarr.Blosc(cname='lz4', clevel=5, shuffle=1)
        encoding = {}

        # Set encoding for data variables
        for var in ds_fixed.data_vars:
            encoding[var] = {
                'chunks': (chunk_sizes['time'], chunk_sizes['lat'], chunk_sizes['lon']),
                'compressor': compressor
            }

        # Set encoding for coordinates
        for coord in ['x', 'y', 'lon', 'lat']:
            if coord in ds_fixed.coords:
                encoding[coord] = {
                    'chunks': chunk_sizes[coord],
                    'compressor': compressor
                }

        self.logger.debug(f"Starting zarr write operation...")
        ds_fixed.to_zarr(output_path, mode='w', encoding=encoding, consolidated=True)
        self.logger.info(f"Saved processed data to {output_path}")
        self.logger.debug(f"... _save_processed_data ===")
    '''
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