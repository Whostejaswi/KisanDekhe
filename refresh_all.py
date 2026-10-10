import json
import pandas as pd
from pipeline import DATA_DIR, END, LOCATIONS, get_location

# Collect every place: built-in towns, anything already cached (including
# custom areas from add_area.py), and every farm in farms.json
targets = {}
for f in DATA_DIR.glob('*.csv'):
    row = pd.read_csv(f, nrows=1).iloc[0]
    targets[row['location']] = (row['lat'], row['lon'])
targets.update(LOCATIONS)
for farm in json.load(open('farms.json', encoding='utf-8')):
    targets[farm['name']] = (farm['lat'], farm['lon'])

print(f'Rebuilding {len(targets)} places up to {END}\n')
for name, (lat, lon) in targets.items():
    try:
        df = get_location(name, lat, lon, refresh=True)
        print(f'{name}: {df["date"].min():%Y-%m-%d} to {df["date"].max():%Y-%m-%d}')
    except Exception as e:
        print(f'{name}: FAILED -> {e}')