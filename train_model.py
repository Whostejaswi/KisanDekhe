import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

HORIZON = 10
CUT = pd.Timestamp('2025-07-01')
NDVI_THR = -0.07
RAIN_THR = 40
SOIL = {'Kinathukadavu': 1, 'Karamadai': 1, 'Sulur': 1}

df = pd.read_csv('gee_data_3locations.csv', parse_dates=['date'])
df = df.sort_values(['location', 'date']).reset_index(drop=True)
df['doy'] = df['date'].dt.dayofyear


def dry_run(s):
    out, c = [], 0
    for v in s:
        c = c + 1 if v < 1 else 0
        out.append(c)
    return out


parts = []
for loc, g in df.groupby('location'):
    g = g.copy()
    g['consecutive_dry_days'] = dry_run(g['rainfall'])
    for w in (7, 14, 30):
        g[f'rain_{w}d'] = g['rainfall'].rolling(w, min_periods=1).sum()
    g['temp_7d'] = g['temp_max'].rolling(7, min_periods=1).mean()
    g['heat_days_7d'] = (g['temp_max'] > 35).astype(int).rolling(7, min_periods=1).sum()
    g['ndvi_change_16d'] = g['ndvi'] - g['ndvi'].shift(16)
    parts.append(g)
df = pd.concat(parts, ignore_index=True)

baselines = []
for loc, g in df[df['date'] < CUT].groupby('location'):
    s = g.groupby('doy')['ndvi'].mean().reindex(range(1, 367)).interpolate(limit_direction='both')
    ext = pd.concat([s.iloc[-15:], s, s.iloc[:15]])
    sm = ext.rolling(15, center=True).mean().iloc[15:-15]
    baselines.append(pd.DataFrame({'location': loc, 'doy': sm.index, 'expected_ndvi': sm.values}))
baseline = pd.concat(baselines, ignore_index=True)
baseline.to_csv('ndvi_baseline.csv', index=False)

df = df.merge(baseline, on=['location', 'doy'], how='left')
df['ndvi_deviation'] = (df['ndvi'] - df['expected_ndvi']) / df['expected_ndvi']
df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365)
df['soil_type'] = df['location'].map(SOIL)

df['stress_now'] = ((df['ndvi_deviation'] < NDVI_THR) & (df['rain_14d'] < RAIN_THR)).astype(int)
df['target'] = df.groupby('location')['stress_now'].transform(
    lambda s: s[::-1].rolling(HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
)
df = df.dropna(subset=['target', 'ndvi_change_16d', 'ndvi_deviation']).copy()

features = ['ndvi_deviation', 'ndvi_change_16d', 'consecutive_dry_days',
            'rain_7d', 'rain_14d', 'rain_30d', 'temp_max', 'temp_7d',
            'heat_days_7d', 'doy_sin', 'doy_cos']
if df['soil_type'].nunique() > 1:
    features.append('soil_type')

print('Overall stress-event rate:', round(df['target'].mean(), 3))
print(df.groupby('location')['target'].mean().round(3))

gap = CUT - pd.Timedelta(days=HORIZON)
train = df[df['date'] < gap]
test = df[df['date'] >= CUT]
print('Train rows:', len(train), '| Test rows:', len(test))
print('Train positives:', int(train['target'].sum()), '| Test positives:', int(test['target'].sum()))

pos = max(train['target'].sum(), 1)
spw = (len(train) - pos) / pos
model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=spw, eval_metric='logloss')
model.fit(train[features], train['target'])

proba = model.predict_proba(test[features])[:, 1]
pred = (proba > 0.5).astype(int)
print('\nXGBoost forecast model')
print(classification_report(test['target'], pred, zero_division=0))
if test['target'].nunique() > 1:
    print('ROC-AUC:', round(roc_auc_score(test['target'], proba), 3))

print('\nPersistence baseline (stressed now -> stressed later)')
print(classification_report(test['target'], test['stress_now'], zero_division=0))

imp = pd.Series(model.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 5))
imp.plot.barh()
plt.title('Feature importance')
plt.tight_layout()
plt.savefig('plot_feature_importance.png', dpi=150)

for loc, g in test.assign(risk=proba).groupby('location'):
    plt.figure(figsize=(9, 3.5))
    plt.plot(g['date'], g['risk'], label='Predicted risk')
    plt.fill_between(g['date'], 0, g['target'], alpha=0.25, label='Actual stress in next 10d')
    plt.title(f'{loc}: risk vs proxy-labelled stress (test period)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plot_risk_{loc}.png', dpi=150)

df['risk'] = model.predict_proba(df[features])[:, 1]
df.to_csv('model_data.csv', index=False)
joblib.dump({'model': model, 'features': features}, 'stress_model.pkl')
print('\nSaved: stress_model.pkl, model_data.csv, plots')