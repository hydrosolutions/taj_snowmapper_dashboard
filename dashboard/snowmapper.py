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
import geoviews as gv
from cartopy import crs
import pandas as pd
import param
from datetime import datetime, timedelta
import numpy as np
import logging
from dotenv import load_dotenv
from typing import Optional

from utils.logging import LoggerSetup
from utils.config import ConfigLoader
from utils.data_warning import DataFreshnessManager

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
    load_dotenv(env_file)

# Load configuration
config_loader = ConfigLoader()
config = config_loader.load_config('local')

# Setup logging
logger_setup = LoggerSetup(config)
logger = logger_setup.setup()

# Color settings
# Set color map for filled contours
# Inspired by https://whiterisk.ch/de/conditions/snow-maps/new_snow
MAP_COLORS_NEW_SNOW = ['#cdffcd', '#99f0b2', '#53bd9f', '#3296b4', '#0670b0', '#054f8c', '#610432', '#4d020f']
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
        'Satellite': gv.tile_sources.EsriImagery,

    }

    def __init__(self, data_dir: Path, config: dict):
        self.data_dir = Path(data_dir)
        self.config = config
        self._cached_data = {}
        self.logger = logging.getLogger('snowmapper.viewer')

        # Get bounds from projections config
        self.bounds = self.config['projections']['bounds']['web_mercator']

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
                # Remove invalid cache entry
                self.logger.debug(f"Removing invalid cache entry for {var_name}")
                del self._cached_data[var_name]

        try:
            # Read data
            if not zarr_path.exists():
                self.logger.error(f"Zarr file does not exist: {zarr_path}")
                return None

            ds = xr.open_zarr(zarr_path)
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

    def create_map(self, var_name: str, time_idx: datetime, data_type: str = 'forecast',
                  basemap: str = 'CartoDB Positron', opacity: float = 0.7) -> gv.Image:
        """Create a map visualization with variable overlay."""
        try:
            # Get base map first
            map_view = self.create_base_map(basemap)

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
            data = ds[var_key].sel(time=time_idx, method='nearest')

            # Make zeros transparent
            data = data.where(data != 0)

            # Create contour levels
            min_val = var_config['min_value']
            max_val = var_config['max_value']
            # Get minimum and maximum values from data
            min_val = data.min().values.item() if np.isfinite(min_val) else var_config['min_value']
            max_val = data.max().values.item() if np.isfinite(max_val) else var_config['max_value']
            n_levels = 10  # Adjust number of contour levels as needed
            levels = np.linspace(min_val, max_val, n_levels)
            self.logger.debug(f"levels for contours: {levels}")

            # Create filled contours (optional)
            filled_contours = hv.QuadMesh((data.lon, data.lat, data)).opts(
                colorbar=True,
                cmap=var_config['colormap'],
                clim=(min_val, max_val),
                alpha=opacity * 0.5,  # Reduce opacity for filled contours
                tools=['hover']
            )

            # Create contour lines
            contours = hv.operation.contours(hv.QuadMesh((data.lon, data.lat, data)), levels=levels).opts(
                line_color='black',
                line_width=1,
                alpha=opacity,
                tools=['hover']
            )

            # Create the raster layer with user-defined opacity
            raster = gv.Image(
                data,
                kdims=['lon', 'lat'],
                vdims=[var_config['name']]
            ).opts(
                colorbar=True,
                cmap=var_config['colormap'],
                clim=(var_config['min_value'], var_config['max_value']),
                tools=['hover'],
                alpha=opacity,
                data_aspect=1,
                show_grid=False,
                title=f"{var_config['name']} ({var_config['units']}) - {pd.to_datetime(time_idx).strftime('%Y-%m-%d')}"
            )

            # Combine base map with raster
            return (map_view * raster).opts(
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
            self.logger.error(f"Error creating map: {e}")
            self.logger.exception("Detailed error:")
            return gv.Text(0, 0, f"Error: {str(e)}")


class SnowMapDashboard(param.Parameterized):
    variable = param.Selector()
    data_type = param.Selector(objects=['forecast', 'accumulated', 'historical'])
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
        variables = ['None'] + list(config['variables'].keys())
        self.param.variable.objects = variables
        params['variable'] = variables[1]  # Start with 'None' selected

        super().__init__(**params)

        # Initialize viewer
        self.viewer = SnowMapViewer(data_dir, config)
        self.data_type = 'forecast'

        # Set up variable selector with None option
        variables = ['None'] + list(config['variables'].keys())
        self.param.variable.objects = variables
        params['variable'] = variables[0]  # Start with 'None' selected

        # Initialize time handling
        self.reference_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self._update_time_bounds()
        #self.data_freshness_manager.update_warning_visibility(self.param.time_offset.bounds, self.config)

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
            min_days = max(self.config['dashboard']['day_slider_min'], min(days_available))
            max_days = min(self.config['dashboard']['day_slider_max'], max(days_available))

            # Update slider bounds
            self.param.time_offset.bounds = (min_days, max_days)

            # Set default to 0 (today) if available, otherwise earliest available day
            if 0 in days_available:
                self.time_offset = 0
            else:
                self.time_offset = min_days

    @param.depends('variable', 'data_type')
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
    name='Variable',
    options={
        dashboard.get_variable_label(var): var
        for var in dashboard.param.variable.objects
    },
    value=dashboard.variable
)

# Create time slider
time_slider = pn.widgets.IntSlider(
    name='Day Offset',
    value=dashboard.time_offset,
    start=dashboard.param.time_offset.bounds[0],
    end=dashboard.param.time_offset.bounds[1],
    step=1
)

# Create map controls
basemap_selector = pn.widgets.RadioButtonGroup(
    name='Base Map',
    options=list(SnowMapViewer.TILE_SOURCES.keys()),
    value='CartoDB Positron'
)

opacity_slider = pn.widgets.FloatSlider(
    name='Layer Opacity',
    value=0.7,
    start=0.1,
    end=1.0,
    step=0.1
)

# Link controls
variable_selector.link(dashboard, value='variable')
basemap_selector.link(dashboard, value='basemap')
opacity_slider.link(dashboard, value='opacity')
time_slider.link(dashboard, value='time_offset')

# Create dynamic control panel
def get_control_panel(variable):
    base_controls = pn.Column(
        pn.pane.Markdown("### Map Controls"),
        variable_selector,
        pn.pane.Markdown("Select base map", margin=(0, 0, -10, 10)), #(top, right, bottom, left)
        basemap_selector,
    )

    if variable != 'None':
        return pn.Column(
            base_controls,
            pn.pane.Markdown("### Variable Controls"),
            dashboard.param.data_type,
            time_slider,
            opacity_slider
        )
    return base_controls

# Create the dashboard layout with dynamic controls
controls = pn.bind(get_control_panel, dashboard.param.variable)

# Initialize template
template = pn.template.BootstrapTemplate(
    title="Snow Situation Tajikistan",
    logo=config['paths']['favicon_path'],
    sidebar_width=350,
    header_background="#2B547E",  # Dark blue header
    favicon=config['paths']['favicon_path']
)

# Add controls to the sidebar
template.sidebar.append(
    controls,
)

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
    pn.pane.Markdown("""
    ### About this Dashboard
    This dashboard shows the snow situation in Tajikistan.
    Data is updated daily and includes:
    - Snow Height (HS) in meters
    - Snow Water Equivalent (SWE) in millimeters
    - Snow melt (SM) in millimeters

    [Close]
    """),
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