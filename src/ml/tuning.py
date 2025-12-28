import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import optuna

from src.ml.preprocessor import CustomerChurnPreprocessor

print("Veri yükleniyor ve işleniyor...")
train_data = pd.read_csv('data/processed/train.csv')

preprocessor = CustomerChurnPreprocessor()
train_data_processed = preprocessor.preprocess(train_data)

X_train = train_data_processed.drop('Exited', axis=1)
y_train = train_data_processed['Exited']

print(f"Eğitim Seti Boyutu: {X_train.shape}")



def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_state': 42
    }
    
    pipeline = make_pipeline(
        VarianceThreshold(threshold=0),
        MinMaxScaler(),
        GradientBoostingClassifier(**params)
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return scores.mean()

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_state': 42,
        'verbose': 0,  
        'allow_writing_files': False
    }
    
    pipeline = make_pipeline(
        VarianceThreshold(threshold=0),
        MinMaxScaler(),
        CatBoostClassifier(**params)
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return scores.mean()

print("1. Gradient Boosting Optimizasyonu Başlıyor...")
study_gb = optuna.create_study(direction='maximize')
study_gb.optimize(objective_gb, n_trials=50) 

print("\n2. CatBoost Optimizasyonu Başlıyor...")
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=50)

print("-" * 50)
print("SONUÇLAR")
print("-" * 50)
print(f"Gradient Boosting En İyi Skor (ROC-AUC): {study_gb.best_value:.4f}")
print("En İyi Parametreler:", study_gb.best_params)
print("-" * 50)
print(f"CatBoost En İyi Skor (ROC-AUC): {study_cat.best_value:.4f}")
print("En İyi Parametreler:", study_cat.best_params)
print("-" * 50)