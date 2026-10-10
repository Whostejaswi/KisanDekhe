from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline import FEATURES, load_all
from advisory import farm_advice

OUT = Path('dashboard')
OUT.mkdir(exist_ok=True)
WINDOW_DAYS = 365
FARM_FIELDS = ('owner', 'crop', 'variety', 'village', 'area_acres', 'sown_on', 'irrigation')

DRIVERS = {
    'ndvi_deviation': ("Crop is less green than normal",
                       "Check leaves for yellowing. Look for pests or nutrient shortage."),
    'ndvi_change_16d': ("Greenness dropped in the last 2 weeks",
                        "Inspect for pests, disease or waterlogging."),
    'consecutive_dry_days': ("Long dry spell", "Plan irrigation now."),
    'rain_7d': ("Little rain this week", "Irrigate if the soil is dry."),
    'rain_14d': ("Little rain in the last 2 weeks", "Irrigate if the soil is dry."),
    'rain_30d': ("Little rain this month", "Review the irrigation schedule."),
    'temp_max': ("Very hot today", "Irrigate early morning or evening. Mulch if possible."),
    'temp_7d': ("Hot week overall", "Keep the soil moist. Avoid midday field work."),
    'heat_days_7d': ("Several days above 35°C", "Keep the soil moist. Watch for afternoon wilting."),
    'doy_sin': ("Crop in a sensitive growth stage", "Check the crop more often this season."),
    'doy_cos': ("Crop in a sensitive growth stage", "Check the crop more often this season."),
}


def level_of(r):
    return 'High' if r >= 0.5 else 'Medium' if r >= 0.2 else 'Low'


model = joblib.load('stress_model.pkl')['model']

df = load_all().dropna(subset=FEATURES).sort_values(['location', 'date']).reset_index(drop=True)
df['risk'] = model.predict_proba(df[FEATURES])[:, 1]

contrib = model.get_booster().predict(xgb.DMatrix(df[FEATURES]), pred_contribs=True)[:, :-1]
reasons = []
for row in contrib:
    labels = []
    for i in np.argsort(-row):
        if row[i] <= 0 or len(labels) == 3:
            break
        label = DRIVERS[FEATURES[i]][0]
        if label not in labels:
            labels.append(label)
    reasons.append(labels)
df['reasons'] = reasons

recent = df.groupby('location', group_keys=False).tail(WINDOW_DAYS)
farms = {f['name']: f for f in json.load(open('farms.json', encoding='utf-8'))}

locations = {}
for name, g in recent.groupby('location'):
    farm = farms.get(name)
    series = []
    for _, r in g.iterrows():
        level = level_of(r['risk'])
        row = {
            'date': r['date'].strftime('%Y-%m-%d'), 'risk': round(float(r['risk']), 3),
            'level': level, 'ndvi': round(float(r['ndvi']), 3),
            'expected': round(float(r['expected_ndvi']), 3),
            'rain': round(float(r['rainfall']), 1), 'tmax': round(float(r['temp_max']), 1),
            'reasons': r['reasons'],
        }
        if farm:
            row['advice'] = farm_advice(farm, r['date'], level)
        series.append(row)

    entry = {'lat': float(g['lat'].iloc[0]), 'lon': float(g['lon'].iloc[0]),
             'is_farm': farm is not None, 'series': series}
    if farm:
        entry.update({k: farm.get(k, '') for k in FARM_FIELDS})
    locations[name] = entry

print('Farms exported:', [n for n, v in locations.items() if v['is_farm']] or 'NONE, run refresh_all.py')

payload = {
    'latest_data': df['date'].max().strftime('%d %b %Y'),
    'metrics': json.load(open('metrics.json')),
    'advice': {label: tip for label, tip in DRIVERS.values()},
    'locations': locations,
}
(OUT / 'data.js').write_text('window.DASHBOARD = ' + json.dumps(payload, ensure_ascii=False) + ';',
                             encoding='utf-8')
print(f'Exported {len(locations)} areas to dashboard/data.js')