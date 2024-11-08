# Description: A simple dashboard to visualize snow height data in Tajikistan.
#
# Usage: panel serve snowmapper.py --show --autoreload

import panel as pn
import holoviews as hv
import geoviews as gv
import xarray as xr
import param
import numpy as np
from holoviews.element.tiles import EsriImagery, OSM
import cartopy.crs as ccrs
import geopandas as gpd

# Initialize extensions
pn.extension('vega', sizing_mode="stretch_width")
hv.extension('bokeh')

# Define translations
TRANSLATIONS = {
    'en': {
        'title': 'Tajikistan Snow Height Dashboard',
        'map_type': 'Map Type',
        'language': 'Language',
        'data_layer': 'Data Layer',
        'street_map': 'Street Map',
        'satellite': 'Satellite',
        'relief': 'Relief',
        'snow_height': 'Snow Height',
        'opacity': 'Layer Opacity',
        'date': 'Date',
    },
    'ru': {
        'title': 'Панель мониторинга снежного покрова Таджикистана',
        'map_type': 'Тип карты',
        'language': 'Язык',
        'data_layer': 'Слой данных',
        'street_map': 'Карта улиц',
        'satellite': 'Спутник',
        'relief': 'Рельеф',
        'snow_height': 'Высота снега',
        'opacity': 'Прозрачность слоя',
        'date': 'Дата',
    }
}

class TajikistanDashboard(param.Parameterized):
    language = param.Selector(objects=['en', 'ru'], default='en')
    map_type = param.Selector(objects=['street_map', 'satellite', 'relief'], default='street_map')
    opacity = param.Magnitude(default=0.7, bounds=(0.0, 1.0))
    date = param.Date()

    def __init__(self, **params):
        super().__init__(**params)
        # Load
        # Load Tajikistan boundaries
        self.taj_bounds = gpd.read_file('static/OSMB-Taj-country-borders.geojson')
        # Convert to holoviews polygon
        self.taj_outline = gv.Polygons(self.taj_bounds.geometry)
        self.load_data()

    def load_data(self):
        """Load and prepare the snow height data"""
        try:
            self.ds = xr.open_dataset('HS_20241107.nc')
            # Clip data to Tajikistan boundaries
            mask = self.create_country_mask()
            self.ds['hs'] = self.ds['hs'].where(mask)
            self.date.default = np.datetime64('2024-11-07')
        except Exception as e:
            print(f"Error loading data: {e}")
            self.ds = None

    def create_country_mask(self):
        """Create a mask for the data based on Tajikistan boundaries"""
        if self.ds is None:
            return None

        # Create a grid of lat/lon points
        lon, lat = np.meshgrid(self.ds.lon, self.ds.lat)
        points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(lon.ravel(), lat.ravel()),
            crs=self.taj_bounds.crs
        )

        # Check which points are within Tajikistan
        mask = points.within(self.taj_bounds.unary_union).values
        return mask.reshape(lon.shape)

    def get_text(self, key):
        """Get translated text"""
        return TRANSLATIONS[self.language][key]

    def get_base_map(self):
        """Get the base map based on selected type"""
        if self.map_type == 'street_map':
            return gv.tile_sources.OSM
        elif self.map_type == 'satellite':
            return gv.tile_sources.EsriImagery
        else:  # relief
            return gv.tile_sources.StamenTerrain

    def get_data_layer(self):
        """Create the snow height data layer"""
        if self.ds is None:
            return None

        # Create snow height overlay
        snow_height = gv.Image(
            (self.ds.lon, self.ds.lat, self.ds.hs.values),
            kdims=['Longitude', 'Latitude'],
            vdims=['Snow Height']
        )

        # Combine with country outline
        return snow_height * self.taj_outline.opts(
            fill_alpha=0,
            line_color='red',
            line_width=2
        )

    @param.depends('language', 'map_type', 'opacity')
    def view(self):
        """Create the main view"""
        # Get base map
        base_map = self.get_base_map()

        # Get data layer
        data_layer = self.get_data_layer()

        # Combine layers
        if data_layer is not None:
            map_view = (base_map * data_layer).opts(
                width=800,
                height=600,
                title=self.get_text('title'),
                xlabel='Longitude',
                ylabel='Latitude'
            )
        else:
            map_view = base_map

        # Set bounds to Tajikistan extent
        bounds = self.taj_bounds.total_bounds
        map_view = map_view.redim.range(
            Longitude=(bounds[0], bounds[2]),
            Latitude=(bounds[1], bounds[3])
        )

        return map_view

    def panel(self):
        """Create the dashboard layout"""
        controls = pn.Column(
            pn.widgets.Select(
                name=self.get_text('language'),
                options={'English': 'en', 'Русский': 'ru'},
                value=self.language,
                param=self.param.language
            ),
            pn.widgets.Select(
                name=self.get_text('map_type'),
                options={
                    self.get_text('street_map'): 'street_map',
                    self.get_text('satellite'): 'satellite',
                    self.get_text('relief'): 'relief'
                },
                value=self.map_type,
                param=self.param.map_type
            ),
            pn.widgets.FloatSlider(
                name=self.get_text('opacity'),
                value=self.opacity,
                start=0,
                end=1,
                step=0.1,
                param=self.param.opacity
            )
        )

        return pn.Column(
            pn.Row(
                controls,
                self.view,
                sizing_mode='stretch_width'
            ),
            sizing_mode='stretch_width'
        )

# Create and show the dashboard
dashboard = TajikistanDashboard()
dashboard.panel().servable()