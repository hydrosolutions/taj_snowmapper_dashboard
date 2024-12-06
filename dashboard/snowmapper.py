# Description: Dashboard for visualizing snow data in Tajikistan.
#
# This script creates a dashboard for visualizing snow data in Tajikistan.
#
# Useage:
# To run the dashboard loaclly, run the following command from the dashboard
# directory:
# panel serve --show snowmapper.py --autoreload
# This will open a new browser window with the dashboard.
#
# Author: Beatrice Marti, hydrosolutions GmbH

import os
import sys
from pathlib import Path

# Add project root to Python path to enable imports from utils
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import xarray as xr
import panel as pn
import holoviews as hv
from holoviews import opts
import geoviews as gv
from cartopy import crs
import pandas as pd
import geopandas as gpd
import param
from datetime import datetime, timedelta
import numpy as np
import logging
from dotenv import load_dotenv
from typing import Optional
from shapely.validation import make_valid
import spatialpandas as spd
import hvplot.pandas
from bokeh.models.tickers import FixedTicker
import pyproj

import gettext

from utils.logging import LoggerSetup
from utils.config import ConfigLoader
from utils.data_warning import DataFreshnessManager

# Printing versions
# Print environment settings
print("PROJ_DATA:", os.environ.get('PROJ_DATA'))
print("PROJ_LIB:", os.environ.get('PROJ_LIB'))

# Print PROJ version info
print("\nPyproj version:", pyproj.__version__)
print("PROJ version:", pyproj.proj_version_str)
print("PROJ data directory:", pyproj.datadir)

# Print GeoPandas info
print("\nGeoPandas version:", gpd.__version__)

# Initialize extensions
pn.extension('tabulator')
hv.extension('bokeh')
gv.extension('bokeh')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('snowmapper')

# Environment setup
env = os.getenv('DASHBOARD_ENV', 'local')
if env == 'aws':
    env_file = '/app/.env'
else:
    env_file = '.env'
if os.path.exists(env_file):
    logger.debug(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)

print(f"Environment: {env}")
print(f"Reading environment variables from: {env_file}")

# Load configuration
config_loader = ConfigLoader()
config = config_loader.load_config(env)

# Setup logging
logger_setup = LoggerSetup(config)
logger = logger_setup.setup()

# Set up translations
lang = config['dashboard']['default_language']
locale_dir = project_root / 'locales'
print(f"Locale dir: {locale_dir}")
loc = gettext.translation('snowmapper', locale_dir, languages=[lang])
print(f"Current language: {lang}")
_ = loc.gettext

# xgettext -o locales/messages.pot dashboard/snowmapper.py
# msginit -i locales/messages.pot -o locales/en/LC_MESSAGES/snowmapper.po -l en
# msginit -i locales/messages.pot -o locales/ru/LC_MESSAGES/snowmapper.po -l ru
# msgfmt -o locales/ru/LC_MESSAGES/snowmapper.mo locales/ru/LC_MESSAGES/snowmapper.po
# msgfmt -o locales/en/LC_MESSAGES/snowmapper.mo locales/en/LC_MESSAGES/snowmapper.po

# Color settings
# Set color map for filled contours
# Inspired by https://whiterisk.ch/de/conditions/snow-maps/new_snow
MAP_COLOR_SCALE_NEW_SNOW = [0.1, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0]  # in cm
MAP_COLOR_SCALE_HS = [0.1, 20.0, 50.0, 80.0, 120.0, 200.0, 300.0, 400.0]  # in cm



# Add this before creating your template
def remove_bokeh_logo(plot, element):
    """Remove the Bokeh logo from the plot."""
    plot.state.toolbar.logo = None


class SnowMapViewer:
    """Handles reading and displaying snow data from Zarr stores."""

    # Define available tile sources
    TILE_SOURCES = {
        'CartoDB Positron': gv.tile_sources.CartoLight,
        #'OpenStreetMap': gv.tile_sources.OSM,
        'Stamen Terrain': gv.tile_sources.StamenTerrain,
        'Satellite': gv.tile_sources.EsriImagery

    }

    def __init__(self, data_dir: Path, config: dict):
        self.data_dir = Path(data_dir)
        self.config = config
        self._cached_data = {}
        self.logger = logging.getLogger('snowmapper.viewer')

        # Get bounds from projections config
        self.bounds = self.config['projections']['bounds']['web_mercator']

        # Get country outline for map, a geojson
        mask_path = project_root / self.config['paths']['mask_path']
        # Replace geojson with shp
        mask_path = mask_path.with_suffix('.shp')
        self.mask_gdf_wgs = gpd.read_file(mask_path)
        self.mask_gdf_wgs = spd.GeoDataFrame(self.mask_gdf_wgs)

    def read_zarr(self, var_name: str) -> Optional[xr.Dataset]:
        """Read Zarr dataset with simple caching."""
        var_name = str(var_name)
        zarr_path = self.data_dir / f"{var_name}_processed.zarr"

        self.logger.debug(f"Attempting to read Zarr file: {zarr_path}")

        # Check cache with validation
        if var_name in self._cached_data:
            timestamp, data = self._cached_data[var_name]
            if (datetime.now() - timestamp).seconds < 3600 and data is not None:  # 1 hour cache
                self.logger.debug(f"Using cached data for {var_name}")
                return data
            else:
                self.logger.debug(f"Removing invalid cache entry for {var_name}")
                del self._cached_data[var_name]

        try:
            if not zarr_path.exists():
                self.logger.error(f"Zarr file does not exist: {zarr_path}")
                return None

            ds = xr.open_zarr(zarr_path)

            # Debug information
            self.logger.debug(f"Raw dataset contents:")
            self.logger.debug(f"Variables: {list(ds.data_vars)}")
            self.logger.debug(f"Coordinates: {list(ds.coords)}")
            self.logger.debug(f"Attributes: {ds.attrs}")

            # Assume Web Mercator if no CRS is found
            import rioxarray
            # If crs is in data variables, extract its attributes and apply them
            if 'crs' in ds.data_vars:
                self.logger.debug(f"CRS variable found with attributes: {ds.crs.attrs}")
                spatial_ref = ds.crs.attrs.get('spatial_ref')
                if spatial_ref:
                    ds.rio.write_crs(spatial_ref, inplace=True)
                else:
                    # Fallback to WG84 if no spatial_ref found
                    ds.rio.write_crs("EPSG:4326", inplace=True)
                    # Fallback to Web Mercator if no spatial_ref found
                    #ds.rio.write_crs("EPSG:3857", inplace=True)
                self.logger.debug(f"Applied CRS: {ds.rio.crs}")

            # Reproject to Web Mercator if needed
            if ds.rio.crs != "EPSG:3857":
                self.logger.debug(f"Reprojecting dataset to Web Mercator")
                ds = ds.rio.reproject("EPSG:3857")

            # Drop the crs variable as it's now in the attributes
            #ds = ds.drop_vars('crs')

            if ds is not None:
                self._cached_data[var_name] = (datetime.now(), ds)
                self.logger.debug(f"Successfully read Zarr file for {var_name}")
                return ds
            else:
                self.logger.error(f"Failed to read data from {zarr_path}")
                return None

        except Exception as e:
            self.logger.error(f"Error reading Zarr file for {var_name}: {e}")
            self.logger.debug(f"Attempted path: {zarr_path}")
            return None

    def get_available_times(self, var_name: str, data_type: str = 'forecast') -> list:
        """Get available time steps for a variable and data type."""
        var_name = str(var_name)
        self.logger.debug(f"Getting available times for {var_name}, type: {data_type}")

        ds = self.read_zarr(var_name)
        if ds is None:
            self.logger.warning(f"No dataset found for variable: {var_name}")
            return []

        try:
            var_key = f"{var_name}_{data_type}"
            if var_key not in ds:
                self.logger.warning(f"Variable {var_key} not found in dataset")
                return []

            times = ds[var_key].time.values
            if times is None or len(times) == 0:
                self.logger.warning(f"No time values found for {var_key}")
                return []

            # Convert numpy datetime64 to Python datetime
            times = [pd.Timestamp(t).to_pydatetime() for t in times]
            self.logger.debug(f"Found {len(times)} timestamps for {var_name}")
            return sorted(times)

        except Exception as e:
            self.logger.error(f"Error getting times for {var_name}: {e}")
            return []

    def create_base_map(self, basemap: str = 'CartoDB Positron') -> gv.Image:
        """Create just the base map without variable overlay."""
        try:
            # Get the appropriate tile source
            tile_source = self.TILE_SOURCES.get(basemap, gv.tile_sources.OSM)

            '''if basemap == 'CartoDB Positron':
                # Create the hillshade layer with reduced opacity
                # Create hillshade layer using ESRI's REST service
                hillshade = gv.WMTS('https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer/tile/{Z}/{Y}/{X}',
                                    crs=crs.GOOGLE_MERCATOR).opts(
                    alpha=0.3,
                    xlabel='',
                    ylabel='',
                )
                tiles = (tile_source() * hillshade).opts(
                    projection=crs.GOOGLE_MERCATOR,
                    global_extent=True
                )
            else:'''
            tiles = tile_source()

            # Set map bounds from config
            return tiles.opts(
                hooks=[remove_bokeh_logo],
                width=1200,  # Allow width to adjust to container
                height=800,  # Allow height to adjust to container
                xaxis=None,  # Remove x axis
                yaxis=None,  # Remove y axis
                active_tools=['pan', 'wheel_zoom'],
                scalebar=True,  # Add scale bar
                xlim=(self.bounds['min_x'], self.bounds['max_x']),
                ylim=(self.bounds['min_y'], self.bounds['max_y']),
                projection=crs.GOOGLE_MERCATOR,
                aspect='equal',
            )

        except Exception as e:
            self.logger.error(f"Error creating base map: {e}")
            self.logger.exception("Detailed error:")
            return gv.Text(0, 0, f"Error: {str(e)}")

    def plot_snow_data(self, data, projection = 'Web Mercator'):
        """Plot snow data using matplotlib."""
        import matplotlib.pyplot as plt
        # Convert dask array to numpy array if needed
        snow_values = data.compute()  # or data.values if already computed

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create mesh grid from lat/lon coordinates
        if projection == 'WGS84':
            self.logger.debug(f"Plotting data in WGS84 projection")
            x, y = np.meshgrid(data.lon, data.lat)
        else:
            self.logger.debug(f"Plotting data in Web Mercator projection")
            x, y = np.meshgrid(data.x, data.y)

        self.logger.debug(f"x: min: {x.min()}, max: {x.max()}")
        self.logger.debug(f"y: min: {y.min()}, max: {y.max()}")

        # Add outline of the country
        if projection == 'WGS84':
            self.mask_gdf_wgs.plot(ax=ax, color='black', linewidth=1)
        else:
            import geopandas as gdp
            mask_gdf_temp = gdp.read_file(config['paths']['mask_path'])
            mask_gdf_web = mask_gdf_temp.to_crs(epsg=3857)
            mask_gdf_web.plot(ax=ax, color='red', linewidth=0.1)

        # Create the plot
        # Using pcolormesh for irregular grids
        im = ax.pcolormesh(x, y, snow_values,
                          shading='auto',
                          cmap='Blues',  # or 'YlOrRd' or any other colormap
                          vmin=0)  # start from 0 for snow height

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)

        # Set labels and title
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.6)

        # Adjust layout
        plt.tight_layout()

        return fig

    def create_label_overrides(self, color_levels, small_threshold=0.01):
        """
        Create major_label_overrides with custom formatting

        Parameters:
        -----------
        color_levels : list
            List of level values
        small_threshold : float
            Threshold below which to use more decimal places
        """
        return {
            level: (
                f'{int(level)}' if level < small_threshold
                else f'{level:.1f}' if level < 10
                else f'{int(level)}'
            )
            for level in color_levels
        }

    def create_custom_colormap(self, var_config, n_colors):
        """
        Create a custom colormap for specific variables
        """
        self.logger.debug(f"Creating custom colormap for {var_config['name']}: {var_config['colormap']}")
        if var_config['colormap'] == 'viridis_r':
            # Create a custom reversed viridis-like colormap
            colors = ['#fde724', '#9fd938', '#49c16d', '#1fa187', '#277e8e',
                      '#365b8c', '#46317e', '#440154'][:n_colors]
        elif var_config['colormap'] == 'YlOrRd':
            # Custom colormap for yellow to red
            colors = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c',
                      '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'][:n_colors]
        # If var_config['colormap'] is a list, use it directly
        elif isinstance(var_config['colormap'], list):
            colors = var_config['colormap'][:n_colors]
        else:
            # Default to a simple blue scale
            colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
                     '#6baed6', '#4292c6', '#2171b5', '#084594'][:n_colors]

        return colors

    def create_contour_plot(self, raster, data_type, var_config, opacity=1.0):
        """
        Create a contour plot using HoloViews with Bokeh backend.

        Parameters
        ----------
        raster : hv.Image or hv.Raster
            The input raster data. Should be a HoloViews Image or Raster object
            containing the data to be plotted. The raster should have:
            - x coordinates
            - y coordinates
            - values (e.g., snow height data)

        data_type : str
            Type of data to be plotted, either 'time_series' or 'accumulated'

        var_config : dict
            Configuration dictionary containing:
            - 'color_levels' : list
                Levels where colors change, e.g., [0.001, 0.2, 0.5, 0.8, 1.2, 2.0, 3.0, 4.0]
            - 'colormap' : str or list
                Either a string naming a colormap (e.g., 'Blues', 'RdYlBu')
                or a list of colors (as hex strings or named colors)
            - 'figure_title' : str
                Title for the colorbar
            - 'units' : str
                Units for the data (appears in colorbar title)

        opacity : float, optional
            Transparency of the contours, between 0 (transparent) and 1 (opaque).
            Default is 1.0.

        Returns
        -------
        hv.Contours
            A HoloViews Contours object that can be displayed or combined with other plots

        Examples
        --------
        var_config = {
            'color_levels': [0.001, 0.2, 0.5, 0.8, 1.2, 2.0, 3.0, 4.0],
            'colormap': 'Blues',  # or ['#ffffff', '#add8e6', '#4169e1', ...]
            'figure_title': 'Snow Height',
            'units': 'm'
        }

        # Create plot
        contour_plot = create_contour_plot(
            raster=snow_height_data,
            var_config=var_config,
            opacity=0.8
        )
        """
        self.logger.debug(f"Creating contour plot")
        self.logger.debug(f"Raster shape: {raster.shape}")

        if data_type == 'time_series':
            # Create a list of colors that is one item shorter than levels
            # as we need n-1 colors for n levels
            levels = var_config['color_levels']

        else:
            levels = var_config['new_snow_color_levels']

        # Set lowest level to 0 if it's very close to zero for better display
        colors = self.create_custom_colormap(var_config, len(levels))
        levels_show = levels
        if levels_show[0] <= 0.005:
            levels_show[0] = 0.0

        # Create a color mapper for the contours
        from bokeh.models import LinearColorMapper
        color_mapper = LinearColorMapper(palette=colors, low=levels[0], high=levels[-1])

        # print debug information
        self.logger.debug(f"Levels: {levels}")
        self.logger.debug(f"Colors: {colors}")
        self.logger.debug(f"type of colors: {type(colors)}")
        self.logger.debug(f"Color mapper: {color_mapper}")

        contours = hv.operation.contours(
            raster,
            levels=levels,  # This defines where the colors change
            filled=True
        ).opts(
            colorbar=True,
            cmap=colors,
            color_levels=levels,  # Explicitly set the levels
            colorbar_opts={
                'ticker': FixedTicker(ticks=levels),
                'major_label_overrides': self.create_label_overrides(levels_show),
                'title': f"{var_config['figure_title']} ({var_config['units']})"
            },
            colorbar_position='top',
            line_color=None,
            line_width=0,
            alpha=opacity,
            width=325,
            height=325,
            #tools=['hover']
        )

        self.logger.debug(f"Contours created successfully")

        return contours

    def create_map(self, var_name: str, time_idx: datetime, data_type: str = 'accumulated',
                   basemap: str = 'CartoDB Positron', opacity: float = 0.7) -> gv.Image:
        """Create a map visualization with variable overlay."""
        self.logger.debug(f"Creating map for {var_name}, {time_idx}, {data_type}, {basemap}, {opacity}")
        try:
            # Get base map first
            map_view = self.create_base_map(basemap)

            # Create a map layer with the country outline
            country_outline = self.mask_gdf_wgs.hvplot(
                geo=True,
                project=True,
                line_color='black',
                line_width=1,
                # No fill
                fill_color=None,
                alpha=1,
            )


            self.logger.debug(f"Reading Zarr data for {var_name}")
            ds = self.read_zarr(var_name)
            if ds is None:
                self.logger.error("Could not read dataset")
                return map_view  # Return just the base map if data can't be loaded

            var_key = f"{var_name}_{data_type}"
            if var_key not in ds:
                self.logger.error(f"Variable {var_key} not found in dataset")
                return map_view  # Return just the base map if variable not found

            # Get variable config
            var_config = self.config['variables'][var_name]

            # Get data for specific time
            self.logger.debug(f"Time index: {time_idx}")
            self.logger.debug(f"Times in ds: {ds.time.values}")
            data = ds[var_key].sel(time=time_idx, method='nearest')

            # Select the first region (assuming region dimension exists)
            if 'region' in data.dims:
                data = data.isel(region=0)

            # Make zeros transparent
            data = data.where(data != 0)

            # Print type of data and data itself
            self.logger.debug(f"Data type: {type(data)}")
            self.logger.debug(f"Data: {data}")
            #self.logger.debug(f"Print part of the data named {var_name}: {data.values}")
            #self.logger.debug(f"Print data which is not nan: {data.values[~np.isnan(data.values)]}")

            # Print loggs for testing if x and y are within self.bounds
            self.logger.debug(f"data min_x: {data.x.min().values.item()}")
            self.logger.debug(f"data max_x: {data.x.max().values.item()}")
            self.logger.debug(f"data min_y: {data.y.min().values.item()}")
            self.logger.debug(f"data max_y: {data.y.max().values.item()}")
            self.logger.debug(f"Map view bounds: {self.bounds}")

            self.logger.debug(f"X coordinate system: {data.x.attrs}")
            self.logger.debug(f"Y coordinate system: {data.y.attrs}")
            self.logger.debug(f"CRS: {data.rio.crs}")
            # If data.rio.crs is not set, set it to Web Mercator
            if data.rio.crs is None:
                data.rio.write_crs("EPSG:3857", inplace=True)
                self.logger.debug(f"Applied CRS: {data.rio.crs}")

            self.logger.debug(f"var_config['name']: {var_config['name']}")

            # if logger mode is set to debug, plot the map using matplotlib and
            # save it as a png file
            #self.logger.debug(f"Logger level: {self.logger.level}")
            #if self.logger.level == 0:
            #    self.logger.debug(f"Plotting map for debugging")
            #    fig = self.plot_snow_data(data, projection='Web Mercator')  # projection: 'Web Mercator' or 'WGS84'
            #    fig.show()
            #    # Create folder debugging if it does not exist
            #    os.makedirs(os.path.join(config['paths']['output_dir'], "debugging"), exist_ok=True)
            #    # Save figure
            #    save_path = f"{config['paths']['output_dir']}/debugging/{var_name}_snowmapper.png"
            #    fig.savefig(save_path)

            self.logger.debug(f"Coordinates in Web Mercator? CRS={data.rio.crs}")
            self.logger.debug(f"X range: {data.x.min().values} to {data.x.max().values}")
            self.logger.debug(f"Y range: {data.y.min().values} to {data.y.max().values}")

            # Create the raster layer with user-defined opacity
            raster = hv.QuadMesh(
                (data.x, data.y, data.values),  # Use x, y coordinates directly
                ['x', 'y'],
                vdims=[var_config['name']]
            )#.opts(
             #   colorbar=True,
             #   cmap=var_config['colormap'],
             #   #clim=(var_config['min_value'], var_config['max_value']),
             #   tools=['hover'],
             #   alpha=opacity,
             #   data_aspect=1,
             #   show_grid=False,
             #   title=f"{var_config['figure_title']} ({var_config['units']}) - {pd.to_datetime(time_idx).strftime('%Y-%m-%d')}"
            #)
            '''
            # Print color levels
            self.logger.debug(f"Color levels: {var_config['color_levels']}")
            contours = hv.operation.contours(
                raster,
                levels=var_config['color_levels'],
                filled=True).opts(
                    colorbar=True,
                    cmap=var_config['colormap'],
                    # Set manual distance between contour lines
                    # Add title to colorbar
                    colorbar_opts={
                        'major_label_overrides': {0.001: '0.001', 0.2: '0.2', 0.5: '0.5', 0.8: '0.8', 1.2: '1.2', 2.0: '2.0', 3.0: '3.0', 4.0: '4.0'},
                        'ticker': FixedTicker(ticks=var_config['color_levels']),
                        'title': f"{var_config['figure_title']} ({var_config['units']})"
                        },
                    # Move the title of the colorbar to the top
                    colorbar_position='top',
                    line_color=None,
                    line_width=0,
                    #tools=['hover'],
                    alpha=opacity,
                    #title=f"{var_config['figure_title']} ({var_config['units']}) - {pd.to_datetime(time_idx).strftime('%Y-%m-%d')}"
                )
            contours.opts(opts.Contours(cmap=var_config['colormap'], colorbar=True, tools=['hover'],
                           width=325, height=325,))
                           #colorbar_opts={'major_label_overrides': {0.001: '0.001', 0.2: '0.2', 0.5: '0.5', 0.8: '0.8', 1.2: '1.2', 2.0: '2.0', 3.0: '3.0', 4.0: '4.0'}}))
            '''
            contours = self.create_contour_plot(raster, data_type, var_config, opacity)

            # Combine base map with raster
            return (map_view * contours * country_outline).opts(
                hooks=[remove_bokeh_logo],
                width=1200,
                height=800,
                #xaxis=None,
                #yaxis=None,
                active_tools=['pan', 'wheel_zoom'],
                scalebar=True,
                xlim=(self.bounds['min_x'], self.bounds['max_x']),
                ylim=(self.bounds['min_y'], self.bounds['max_y']),
                projection=crs.GOOGLE_MERCATOR,
                aspect='equal',
            )

        except Exception as e:
            self.logger.error(f"Error creating map: {e}")
            self.logger.exception("Detailed error:")
            return gv.Text(0, 0, f"Error: {str(e)}")


class SnowMapDashboard(param.Parameterized):
    DATA_TYPE_MAPPING = {
        'time_series': 'Снежная обстановка',  # 'Snow situation',  # key is internal value, value is translation key
        'accumulated': 'Свежий снег'  # 'New snow'
    }
    variable = param.Selector()
    data_type = param.Selector()
    time_offset = param.Integer(default=0, bounds=(config['dashboard']['day_slider_min'], config['dashboard']['day_slider_max']))  # Slider for relative days
    basemap = param.Selector(default='CartoDB Positron', objects=[
        #'OpenStreetMap',
        'Stamen Terrain',
        'Satellite',
        'CartoDB Positron'
    ])
    opacity = param.Number(
        default=config['dashboard']['default_opacity'],
        bounds=(0.1, 1.0),
        step=0.1,
        doc="Opacity of the overlay layer"
    )

    def __init__(self, data_dir: Path, config: dict, **params):
        self.config = config
        self.logger = logging.getLogger('snowmapper.dashboard')

        # Initialize data freshness manager
        self.data_freshness_manager = DataFreshnessManager()

        # Set up variable selector with None option
        #variables = ['None'] + list(config['variables'].keys())
        variables = list(config['variables'].keys())
        self.param.variable.objects = variables
        params['variable'] = variables[0]  # Start with 'None' selected

        super().__init__(**params)

        # Initialize viewer
        self.viewer = SnowMapViewer(data_dir, config)
        self.data_type = 'accumulated'
        self.param.data_type.objects = list(self.DATA_TYPE_MAPPING.keys())

        # Initialize time handling
        self.reference_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self._update_time_bounds()
        #self.data_freshness_manager.update_warning_visibility(self.param.time_offset.bounds, self.config)

    def get_date_label(offset: int) -> str:
        date = dashboard.reference_date + timedelta(days=offset)
        return date.strftime("%a")

    @param.depends('data_type')
    def _update_time_bounds(self):
        """Update time slider bounds based on data availability."""
        if self.variable == 'None':
            return

        var_name = str(self.variable)
        times = self.viewer.get_available_times(var_name, self.data_type)

        if not times:
            self.logger.warning("No times available")
            return

        # Sort times
        times = sorted(times)

        # Calculate relative days from reference date
        days_available = [(t - self.reference_date).days for t in times]

        # Store the time bounds
        self.time_bounds = [days_available[0], days_available[-1]]

        # Always update warning visibility with the current bounds
        self.logger.debug(f"Current time bounds: {self.time_bounds}")
        self.logger.debug(f"Reference date: {self.reference_date}")
        self.logger.debug(f"Days available: {days_available}")

        self.data_freshness_manager.update_warning_visibility(
            self.time_bounds,
            self.config
        )

        # If there is no overlap of days available with the slider bounds,
        # it means we have no data for the current time offset.
        # In this case, we can't display a map.
        if not any([self.time_offset in range(min(days_available), max(days_available))]):
            self.data_freshness_manager.set_warning_visibility(True)
            self.time_offset = 0
            return

        if days_available:
            # If data_type is 'time_series', do the below to get the min and max days
            self.logger.debug(f"\n\n\nData type: {self.data_type}")
            if self.data_type == 'time_series':
                min_days = max(self.config['dashboard']['day_slider_min'], min(days_available))
                max_days = min(self.config['dashboard']['day_slider_max'], max(days_available))

                # Update slider bounds
                self.param.time_offset.bounds = (min_days, max_days)

                # Set default to 0 (today) if available, otherwise earliest available day
                if 0 in days_available:
                    self.time_offset = 0
                else:
                    self.time_offset = min_days
            else:
                if 1 in days_available:
                    self.param.time_offset.bounds = (1, max(days_available))
                    self.time_offset = 1
                else:
                    self.param.time_offset.bounds = (min_days, max(days_available))
                    self.time_offset = min_days
            self.logger.debug(f"Time offset bounds: {self.param.time_offset.bounds}")
            self.logger.debug(f"Time offset: {self.time_offset}")

    def update_time_options(self):
        """Update time options when variable or data type changes."""
        if self.variable != 'None':
            self._update_time_bounds()

    def get_current_time(self) -> datetime:
        """Get the actual datetime based on the current offset."""
        return self.reference_date + timedelta(days=self.time_offset)

    @param.depends('variable', 'data_type', 'time_offset', 'basemap', 'opacity')
    def view(self):
        """Create the map view."""
        logger.debug(f"Creating map view for {self.variable}, {self.data_type}, {self.time_offset}")
        if self.variable == 'None':
            # Return only the basemap without variable overlay
            return self.viewer.create_base_map(self.basemap)
        else:
            # Return map with variable overlay
            var_name = str(self.variable)
            current_time = self.get_current_time()
            return self.viewer.create_map(
                var_name,
                current_time,
                self.data_type,
                self.basemap,
                self.opacity
            )

    def get_data_type_label(self, data_type: str) -> str:
        """Get translated data type label."""
        print(f"\n\nData type: {data_type}")
        print(f"Data type mapping: {self.DATA_TYPE_MAPPING}")
        print(f"Data type label: {self.DATA_TYPE_MAPPING[data_type]}")
        return self.DATA_TYPE_MAPPING[data_type]

    def get_variable_label(self, var_name: str) -> str:
        """Get formatted variable label from config."""
        if var_name == 'None':
            return 'No variable overlay'
        var_config = self.config['variables'][var_name]
        return f"{var_config['widget_short_name']} ({var_config['units']})"


# Initialize the dashboard with proper variable handling
dashboard = SnowMapDashboard(
    data_dir=Path(config['paths']['output_dir']),
    config=config
)

# Create variable selector
variable_selector = pn.widgets.Select(
    name=_('Variable'),
    options={
        dashboard.get_variable_label(var): var
        for var in dashboard.param.variable.objects
    },
    value=dashboard.variable
)

# Create data type selector
data_type_selector = pn.widgets.Select(
    name = _('Data Type'),
    options={
        dashboard.get_data_type_label(data_type): data_type
        for data_type in dashboard.param.data_type.objects
    },
    value = dashboard.data_type
)
print(f"Variable selector options: {variable_selector.options}")
print(f"Data type selector options: {data_type_selector.options}")

# Create time slider
time_slider = pn.widgets.IntSlider(
    name=f'Смещение по дням от {dashboard.reference_date.strftime("%Y-%m-%d")}',  # Day Offset from
    value=dashboard.time_offset,
    start=dashboard.param.time_offset.bounds[0],
    end=dashboard.param.time_offset.bounds[1],
    step=1
)

# Create map controls
basemap_selector = pn.widgets.RadioButtonGroup(
    name=_('Base Map'),
    options=list(SnowMapViewer.TILE_SOURCES.keys()),
    value='CartoDB Positron'
)

opacity_slider = pn.widgets.FloatSlider(
    name=_('Layer Opacity'),
    value=0.7,
    start=0.1,
    end=1.0,
    step=0.1
)

# Link controls
variable_selector.link(dashboard, value='variable')
data_type_selector.link(dashboard, value='data_type')
basemap_selector.link(dashboard, value='basemap')
opacity_slider.link(dashboard, value='opacity')
time_slider.link(dashboard, value='time_offset')

# Create dynamic control panel
def get_control_panel(variable):
    base_controls = pn.Column(
        pn.pane.Markdown(_("### Map Controls")),
        variable_selector,
        pn.pane.Markdown(_("Select base map"), margin=(0, 0, -10, 10)), #(top, right, bottom, left)
        basemap_selector,
    )

    if variable != 'None':
        return pn.Column(
            base_controls,
            pn.pane.Markdown(_("### Variable Controls")),
            data_type_selector,
            time_slider,
            opacity_slider
        )
    return base_controls

# Create the dashboard layout with dynamic controls
controls = pn.bind(get_control_panel, dashboard.param.variable)

# Initialize template
template = pn.template.BootstrapTemplate(
    title=_("Snow Situation Tajikistan"),
    logo=config['paths']['favicon_path'],
    sidebar_width=350,
    header_background="#2B547E",  # Dark blue header
    favicon=config['paths']['favicon_path']
)

# Add controls to the sidebar
template.sidebar.append(
    controls,
)
'''
# Add spacer to push logos to bottom
template.sidebar.append(pn.Spacer(height=80))

# Create logo grid with flexbox layout
logo_grid = pn.FlexBox(
    sizing_mode='stretch_width',
    align='start',
    margin=(0, 10, 10, 10),  # top, right, bottom, left
    styles={
        'display': 'flex',
        'flex-wrap': 'wrap',
        'gap': '10px',
        'justify-content': 'flex-start'  # 'center'
    }
)

# Add each logo from the static/logos directory
logo_dir = config['paths']['favicon_path']
# Discard the filename from the logo_dir and append logos
logo_dir = Path(logo_dir).parent
logo_dir = Path(logo_dir).joinpath('logos')
if logo_dir.exists():
    for logo_file in sorted(logo_dir.glob('*')):
        if logo_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
            logo_grid.append(
                pn.pane.Image(
                    str(logo_file),
                    height=240,  # Smaller height since they're side by side
                    width=None,  # Auto width to maintain aspect ratio
                    sizing_mode='fixed',
                    align='start',
                    styles={
                        #'min-width': '60px',  # Minimum width for each logo
                        #'max-width': '120px',  # Maximum width for each logo
                        'object-fit': 'contain',
                        'margin-left': '10px'
                    }
                )
            )

# Add logo grid to sidebar
template.sidebar.append(logo_grid)
'''

# Add custom CSS for maximizing map space
template.config.raw_css.append("""
.bk-root {
    width: 100%;
    height: 100%;
}

.main-content {
    height: calc(100vh - 50px);
    width: 100%;
    padding: 0 !important;
    margin: 0 !important;
    display: flex;
    flex-direction: column;
}

.bk-root .bk {
    flex-grow: 1;
}
""")

# Create map pane to handle map sizing
map_pane = pn.pane.HoloViews(
    dashboard.view,
    sizing_mode='stretch_both',
    min_height=300,
)

# Add main view to the main area
template.main.append(
    pn.Column(
        dashboard.data_freshness_manager.get_warning_component(),
        map_pane,
        sizing_mode='stretch_both',
        margin=10,
        css_classes=['main-content']
    )
)

# Add custom CSS for floating info
template.config.raw_css.append("""
.floating-info {
    position: absolute;
    top: 60px;
    right: 20px;
    z-index: 1000;
    background: white;
    border: 1px solid #ddd;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
""")

# Create header with date
header = pn.pane.Markdown(
    "",
)

# Create info content
info_content = pn.Column(
    pn.pane.Markdown(_("""
    ### About this Dashboard
    This dashboard shows the snow situation in Tajikistan.
    Data is updated daily and includes:
    - Snow Height (HS) in meters
    - Snow Water Equivalent (SWE) in millimeters

    To view daily accumulated forecasts of new snow, select the 'New snow' data type.
    To view the simulated snow situation (past and forecast), select the 'Snow situation' data type.

    [Close]
    """)),
    width=400,
    css_classes=['floating-info', 'p-3'],
    visible=False
)

# Create info button
info_button = pn.widgets.Button(
    name='ℹ️',
    button_type='primary',
    align='end',
    #sizing_mode='fixed',
    width=50
)

# Toggle info visibility
def toggle_info(event):
    info_content.visible = not info_content.visible

info_button.on_click(toggle_info)

# Add components to template
template.header.append(
    pn.Row(
        header,
        info_button,
        sizing_mode='stretch_width'
    )
)

template.main.append(info_content)


# Make the dashboard servable
template.servable()