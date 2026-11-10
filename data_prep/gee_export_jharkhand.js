// Google Earth Engine script (paste into code.earthengine.google.com, then press Run and start the Export tasks).
// Exports 9 GeoTIFF layers for Jharkhand on ONE common grid, matching the layer names used by the Python pipeline.
// NOT included (do these in QGIS / offline): dist_road, dist_grid (from OpenStreetMap), income (from Census).

var SCALE = 1000;            // metres per cell. Start at 1000 (about 380 x 400 cells). 100 is very heavy for the Python models.
var CRS = 'EPSG:32645';      // UTM zone 45N, metric, suits Jharkhand
var FOLDER = 'jharkhand_layers';

// ---- Study area
var jh = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM1_NAME', 'Jharkhand'));
var region = jh.geometry().bounds();
Map.centerObject(jh, 7); Map.addLayer(jh, {}, 'Jharkhand');

// ---- Solar and climate (ERA5-Land, about 11 km native resolution; mean of 2015-2024)
var era = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate('2015-01-01', '2025-01-01');
var ghi = era.select('surface_solar_radiation_downwards_sum').sum().divide(10).divide(3.6e6).rename('ghi');   // kWh/m2/year
var temp = era.select('temperature_2m').mean().subtract(273.15).rename('temp');                                 // deg C

// ---- Terrain (SRTM 30 m)
var srtm = ee.Image('USGS/SRTMGL1_003');
var elevation = srtm.rename('elevation');                       // metres
var slope = ee.Terrain.slope(srtm).rename('slope');            // degrees

// ---- Land cover (ESA WorldCover 2021) remapped to the pipeline's codes:
// 0 water, 1 forest, 2 barren/grass/shrub, 3 cropland, 4 built-up
var wc = ee.Image('ESA/WorldCover/v200/2021').select('Map');
var lulc = wc.remap([10,20,30,40,50,60,70,80,90,95,100], [1,2,2,3,4,2,2,0,0,1,2]).rename('lulc');

// ---- Vegetation (Sentinel-2 NDVI, median of 2024)
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filterDate('2024-01-01', '2025-01-01')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));
var ndvi = s2.map(function (i) { return i.normalizedDifference(['B8', 'B4']); }).median().rename('ndvi');

// ---- Flood proxy: JRC surface-water occurrence (0 to 1). Replace with a proper flood-hazard layer if you find one.
var flood = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).divide(100).rename('flood');

// ---- Protected areas (WDPA): 1 inside, 0 outside
var wdpa = ee.FeatureCollection('WCMC/WDPA/current/polygons').filterBounds(region);
var protectedArea = ee.Image(0).byte().paint(wdpa, 1).rename('protected');

// ---- Income proxy: night-time lights (replace with Census-based data when you have it). Scaled 0 to 1.
var lights = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').filterDate('2024-01-01', '2025-01-01')
              .select('avg_rad').mean();
var income = lights.unitScale(0, 20).clamp(0, 1).rename('income');

// ---- Inside-Jharkhand mask (1 inside the state boundary)
var inside = ee.Image(0).byte().paint(jh, 1).rename('inside');

// ---- Export every layer on the same grid (same region, CRS and scale)
var layers = {ghi: ghi, temp: temp, elevation: elevation, slope: slope, lulc: lulc, ndvi: ndvi,
              flood: flood, protected: protectedArea, income: income, inside: inside};
Object.keys(layers).forEach(function (name) {
  Export.image.toDrive({
    image: layers[name].toFloat(), description: name, folder: FOLDER, fileNamePrefix: name,
    region: region, crs: CRS, scale: SCALE, maxPixels: 1e10, fileFormat: 'GeoTIFF'
  });
});
