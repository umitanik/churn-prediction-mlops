import os
import sys
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score
from catboost import CatBoostClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.preprocessor import CustomerChurnPreprocessor

best_params_catboost = {
    'iterations': 505,
    'learning_rate': 0.025232339470106814,
    'depth': 4,
    'l2_leaf_reg': 0.4487280568776956,
    'border_count': 87,
    'subsample': 0.727617583882814,
    'random_state': 42,
    'verbose': 0,               
    'allow_writing_files': False
}


print(" Final Model Eğitimi Başlıyor (CatBoost)...")

print(" Veriler yükleniyor...")
train_data = pd.read_csv('data/processed/train.csv')
test_data = pd.read_csv('data/processed/test.csv')
ground_truth = pd.read_csv('data/processed/ground_truth.csv')
preprocessor = CustomerChurnPreprocessor()

train_data_processed = preprocessor.preprocess(train_data)
test_data_processed = preprocessor.preprocess(test_data)

X_train = train_data_processed.drop('Exited', axis=1)
y_train = train_data_processed['Exited']

X_test = test_data_processed
y_test = ground_truth['Exited']

final_pipeline = make_pipeline(
    VarianceThreshold(threshold=0),
    MinMaxScaler(),
    CatBoostClassifier(**best_params_catboost)
)

final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)
y_prob = final_pipeline.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, y_prob)

print("\n" + "="*40)
print(f" FINAL ROC-AUC SKORU: {roc_score:.4f}")
print("="*40)

print(classification_report(y_test, y_pred))

model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'catboost_churn_model.pkl')
joblib.dump(final_pipeline, model_path)
print(f"\n Model ve Pipeline başarıyla kaydedildi: {model_path}")
