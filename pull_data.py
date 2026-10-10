import ee
import pandas as pd

ee.Initialize(project='nodal-seer-478715-m2')

points = {
    'Kinathukadavu': (10.8163, 77.0153),
    'Karamadai': (11.1840, 76.9330),
    'Sulur': (11.0240, 77.1270),
}

start, end = '2024-01-01', '2026-06-01'

chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(start, end)
era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(start, end)
modis = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(start, end)


def extract(collection, band, scale, point, name, col):
    def f(img):
        val = img.reduceRegion(ee.Reducer.mean(), point, scale).get(band)
        return ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), col: val})
    fc = collection.select(band).map(f).getInfo()
    df = pd.DataFrame([x['properties'] for x in fc['features']])
    df['location'] = name
    return df.dropna()


frames = []
for name, (lat, lon) in points.items():
    print('Pulling', name)
    pt = ee.Geometry.Point(lon, lat)
    rain = extract(chirps, 'precipitation', 5000, pt, name, 'rainfall')
    temp = extract(era5, 'temperature_2m_max', 11000, pt, name, 'temp_max')
    ndvi = extract(modis, 'NDVI', 250, pt, name, 'ndvi')

    temp['temp_max'] = temp['temp_max'] - 273.15
    ndvi['ndvi'] = ndvi['ndvi'] * 0.0001

    df = rain.merge(temp, on=['date', 'location'], how='inner')
    df['date'] = pd.to_datetime(df['date'])
    ndvi['date'] = pd.to_datetime(ndvi['date'])

    df = df.merge(ndvi[['date', 'ndvi']], on='date', how='left')
    df = df.sort_values('date')
    df['ndvi'] = df['ndvi'].interpolate().bfill().ffill()
    frames.append(df)

final = pd.concat(frames, ignore_index=True)
final.to_csv('gee_data_3locations.csv', index=False)
print(final.groupby('location').size())
print(final.head())