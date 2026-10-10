import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

bundle = joblib.load('stress_model.pkl')
model, features = bundle['model'], bundle['features']

df = pd.read_csv('model_data.csv', parse_dates=['date'])
test = df[df['date'] >= '2025-07-01'].copy()
test['risk'] = model.predict_proba(test[features])[:, 1]

print('All test days')
for t in (0.2, 0.3, 0.4, 0.5, 0.6):
    p = (test['risk'] > t).astype(int)
    print(t, 'precision', round(precision_score(test['target'], p, zero_division=0), 2),
          'recall', round(recall_score(test['target'], p), 2))

onset = test[test['stress_now'] == 0]
print('\nOnset days (not stressed today):', len(onset),
      '| stress follows within 10d:', int(onset['target'].sum()))
if onset['target'].nunique() > 1:
    print('Onset ROC-AUC:', round(roc_auc_score(onset['target'], onset['risk']), 3))
for t in (0.2, 0.3, 0.4, 0.5):
    p = (onset['risk'] > t).astype(int)
    print(t, 'precision', round(precision_score(onset['target'], p, zero_division=0), 2),
          'recall', round(recall_score(onset['target'], p), 2))
print('Persistence on onset days: recall 0.00 by construction')