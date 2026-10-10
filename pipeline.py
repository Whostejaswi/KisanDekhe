"""Google Earth Engine -> daily features per location. Same science as train_model.py."""
import re
from pathlib import Path

import ee
import numpy as np
import pandas as pd

PROJECT = 'nodal-seer-478715-m2'
START, END = '2024-01-01', '2026-09-30'
CUT = pd.Timestamp('2025-07-01')        # train/test split; baseline uses data before this
HORIZON = 10
NDVI_THR, RAIN_THR = -0.10, 20

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

FEATURES = ['ndvi_deviation', 'ndvi_change_16d', 'consecutive_dry_days',
            'rain_7d', 'rain_14d', 'rain_30d', 'temp_max', 'temp_7d',
            'heat_days_7d', 'doy_sin', 'doy_cos']

# Approximate coordinates: verify on Google Maps before the demo
LOCATIONS = {
    'Kinathukadavu': (10.8163, 77.0153),
    'Karamadai': (11.1840, 76.9330),
    'Sulur': (11.0240, 77.1270),
    'Coimbatore': (11.0168, 76.9558),
    'Tiruppur': (11.1085, 77.3411),
    'Avinashi': (11.1946, 77.2687),
    'Perundurai': (11.2750, 77.5833),
    'Erode': (11.3410, 77.7172),
    'Dharapuram': (10.7356, 77.5253),
    'Namakkal': (11.2189, 78.1674),
    'Salem': (11.6643, 78.1460),
}

_ee_ready = False


def init_ee():
    global _ee_ready
    if not _ee_ready:
        ee.Initialize(project=PROJECT)
        _ee_ready = True


def _extract(collection, band, scale, point, col, factor=1.0, offset=0.0):
    def per_image(img):
        val = img.reduceRegion(ee.Reducer.mean(), point, scale).get(band)
        return ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), col: val})

    fc = collection.select(band).map(per_image).getInfo()
    df = pd.DataFrame([f['properties'] for f in fc['features']]).dropna(subset=[col])
    df['date'] = pd.to_datetime(df['date'])
    df[col] = df[col].astype(float) * factor + offset
    return df[['date', col]]


def fetch_raw(lat, lon):
    init_ee()
    pt = ee.Geometry.Point(lon, lat)
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(START, END)
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(START, END)
    modis = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(START, END)

    rain = _extract(chirps, 'precipitation', 5000, pt, 'rainfall')
    temp = _extract(era5, 'temperature_2m_max', 11000, pt, 'temp_max', 1.0, -273.15)
    ndvi = _extract(modis, 'NDVI', 250, pt, 'ndvi', 0.0001)

    df = rain.merge(temp, on='date', how='inner').sort_values('date')
    df = df.merge(ndvi, on='date', how='left').sort_values('date')
    df['ndvi'] = df['ndvi'].interpolate().bfill().ffill()
    return df.reset_index(drop=True)


def _baseline(df):
    """Expected NDVI per day-of-year, learned only from data before CUT."""
    hist = df[df['date'] < CUT]
    s = hist.groupby('doy')['ndvi'].mean().reindex(range(1, 367)).interpolate(limit_direction='both')
    ext = pd.concat([s.iloc[-15:], s, s.iloc[:15]]).reset_index(drop=True)
    smooth = ext.rolling(15, center=True).mean().iloc[15:-15].values
    return pd.Series(smooth, index=range(1, 367))


def add_features(raw):
    df = raw.sort_values('date').reset_index(drop=True).copy()
    df['doy'] = df['date'].dt.dayofyear

    dry, c = [], 0
    for v in df['rainfall']:
        c = c + 1 if v < 1 else 0
        dry.append(c)
    df['consecutive_dry_days'] = dry

    for w in (7, 14, 30):
        df[f'rain_{w}d'] = df['rainfall'].rolling(w, min_periods=1).sum()
    df['temp_7d'] = df['temp_max'].rolling(7, min_periods=1).mean()
    df['heat_days_7d'] = (df['temp_max'] > 35).astype(int).rolling(7, min_periods=1).sum()
    df['ndvi_change_16d'] = df['ndvi'] - df['ndvi'].shift(16)

    df['expected_ndvi'] = df['doy'].map(_baseline(df))
    df['ndvi_deviation'] = (df['ndvi'] - df['expected_ndvi']) / df['expected_ndvi']
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365)

    df['stress_now'] = ((df['ndvi_deviation'] < NDVI_THR) & (df['rain_14d'] < RAIN_THR)).astype(int)
    # label = 1 if stress occurs in the next HORIZON days (NaN for the last days: used for the live view)
    df['target'] = (df['stress_now'][::-1].rolling(HORIZON, min_periods=HORIZON)
                    .max()[::-1].shift(-1))
    return df


def slugify(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def get_location(name, lat, lon, refresh=False):
    """Return cached features for a place, fetching from GEE if not cached."""
    path = DATA_DIR / f'{slugify(name)}.csv'
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=['date'])
    df = add_features(fetch_raw(lat, lon))
    df.insert(0, 'location', name)
    df['lat'], df['lon'] = lat, lon
    df.to_csv(path, index=False)
    return df


def load_all():
    files = sorted(DATA_DIR.glob('*.csv'))
    return pd.concat([pd.read_csv(f, parse_dates=['date']) for f in files], ignore_index=True)