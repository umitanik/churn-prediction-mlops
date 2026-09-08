import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import catboost
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data.split_dataset import PROCESSED_DIR, ensure_split
from src.ml.preprocessor import CustomerChurnPreprocessor, build_encoder


def load_training_data(processed_dir=PROCESSED_DIR):
    print("Loading and preprocessing data...")
    train_path, _, _ = ensure_split(processed_dir=processed_dir)
    train_data = pd.read_csv(train_path)

    preprocessor = CustomerChurnPreprocessor()
    train_data_processed = preprocessor.preprocess(train_data)

    X_train = train_data_processed.drop('Exited', axis=1)
    y_train = train_data_processed['Exited']

    print(f"Training Set Size: {X_train.shape}")
    return X_train, y_train


def benchmark_models(X, y):
    estimators = {
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "CatBoost": catboost.CatBoostClassifier(verbose=0, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(verbose=-1, random_state=42),
        "Naive Bayes": GaussianNB()
    }

    # Every candidate gets the same fitted encoder and scaler, so the
    # comparison reflects the estimator rather than the preprocessing.
    models = {
        name: make_pipeline(build_encoder(), MinMaxScaler(), estimator)
        for name, estimator in estimators.items()
    }

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    print(f"\n{len(models)} Different Models are being tested on Train Set with 5-Fold CV...\n")
    print(f"{'Model Name':<25} | {'ROC-AUC Score'}")

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


def plot_results(results_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x="ROC-AUC Mean", y="Model", data=results_df, palette="viridis", hue="Model", legend=False)
    plt.title("Model Comparison (Train Set Only - 5-Fold CV)")
    plt.xlabel("Average ROC-AUC Score")
    plt.xlim(0.5, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    X_train, y_train = load_training_data()
    results_df = benchmark_models(X_train, y_train)

    plot_results(results_df)

    print(results_df)

    best_model_name = results_df.iloc[0:3]['Model']
    print(f"\nTop performing models based on ROC-AUC:\n{best_model_name}")


if __name__ == "__main__":
    main()
