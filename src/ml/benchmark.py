import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import lightgbm as lgb
import xgboost as xgb
import catboost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.ml.preprocessor import CustomerChurnPreprocessor

print("Veri yükleniyor ve işleniyor...")
train_data = pd.read_csv('data/processed/train.csv')

preprocessor = CustomerChurnPreprocessor()
train_data_processed = preprocessor.preprocess(train_data)

X_train = train_data_processed.drop('Exited', axis=1)
y_train = train_data_processed['Exited']

print(f"Eğitim Seti Boyutu: {X_train.shape}")


def benchmark_models(X, y):
    models = {
        "K-Nearest Neighbors": make_pipeline(MinMaxScaler(), KNeighborsClassifier()),
        "Support Vector Machine": make_pipeline(MinMaxScaler(), SVC(probability=True, random_state=42)),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "CatBoost": catboost.CatBoostClassifier(verbose=0, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(verbose=-1, random_state=42),
        "Naive Bayes": GaussianNB()
    }
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    print(f"\n{len(models)} Farklı Model 5-Fold CV ile Train Seti Üzerinde Test Ediliyor...\n")
    print(f"{'Model Adı':<25} | {'ROC-AUC Skoru'}")
    
    for name, model in models.items():
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc', n_jobs=-1)
        
        results.append({
            "Model": name,
            "ROC-AUC Mean": cv_results.mean(),  
            "ROC-AUC Std": cv_results.std() 
        })
        
        print(f" {name:<25} | {cv_results.mean():.4f} (+/- {cv_results.std():.4f})")
        
    df_results = pd.DataFrame(results).sort_values(by="ROC-AUC Mean", ascending=False)
    return df_results

results_df = benchmark_models(X_train, y_train)

plt.figure(figsize=(12, 6))
sns.barplot(x="ROC-AUC Mean", y="Model", data=results_df, palette="viridis", hue="Model", legend=False)
plt.title("Model Karşılaştırması (Yalnızca Train Seti - 5-Fold CV)")
plt.xlabel("Ortalama ROC-AUC Skoru")
plt.xlim(0.5, 1.0) 
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print(results_df)

best_model_name = results_df.iloc[0:3]['Model']
print(f"\n {best_model_name} modeli en yüksek skoru verdi.")
