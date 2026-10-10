import json
from pathlib import Path

import pandas as pd

print('--- Files ---')
for p in ['pipeline.py', 'advisory.py', 'farms.json', 'stress_model.pkl',
          'metrics.json', 'dashboard/index.html', 'dashboard/data.js']:
    print(('OK      ' if Path(p).exists() else 'MISSING ') + p)

print('\n--- Cached places ---')
files = sorted(Path('data').glob('*.csv'))
print(f'{len(files)} CSV files in data/')
for f in files:
    df = pd.read_csv(f, parse_dates=['date'])
    print(f'{f.stem:35s} {df["date"].min():%Y-%m-%d} to {df["date"].max():%Y-%m-%d}  rows={len(df)}')

print('\n--- data.js ---')
p = Path('dashboard/data.js')
if p.exists():
    txt = p.read_text(encoding='utf-8')
    print('size:', len(txt), 'bytes')
    print('starts with:', txt[:40])
    print('contains farms:', '"is_farm":true' in txt)
    print('latest data:', json.loads(txt[txt.index('{'):txt.rindex('}') + 1])['latest_data'])