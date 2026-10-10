import json

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from pipeline import CUT, FEATURES, HORIZON, load_all

df = load_all().dropna(subset=FEATURES)
labelled = df.dropna(subset=['target'])
train = labelled[labelled['date'] < CUT - pd.Timedelta(days=HORIZON)]
test = labelled[labelled['date'] >= CUT]

pos = train['target'].sum()
model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=(len(train) - pos) / max(pos, 1),
                          eval_metric='logloss')
model.fit(train[FEATURES], train['target'])

test = test.assign(risk=model.predict_proba(test[FEATURES])[:, 1])
onset = test[test['stress_now'] == 0]
pred = (onset['risk'] > 0.2).astype(int)

metrics = {
    'locations': sorted(df['location'].unique().tolist()),
    'train_rows': len(train),
    'test_rows': len(test),
    'roc_auc_all_days': round(roc_auc_score(test['target'], test['risk']), 3),
    'onset_days': len(onset),
    'onset_positive_days': int(onset['target'].sum()),
    'onset_recall': round(recall_score(onset['target'], pred), 3),
    'onset_precision': round(precision_score(onset['target'], pred, zero_division=0), 3),
    'onset_roc_auc': round(roc_auc_score(onset['target'], onset['risk']), 3),
    'alert_threshold': 0.2,
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
joblib.dump({'model': model, 'features': FEATURES}, 'stress_model.pkl')

print(json.dumps(metrics, indent=2))