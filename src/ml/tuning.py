import logging
import os

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

from src.data.split_dataset import PROCESSED_DIR, ensure_split
from src.ml.preprocessor import CustomerChurnPreprocessor, build_encoder

N_TRIALS = 50


def load_training_data(processed_dir=PROCESSED_DIR):
    logger.info("Loading and preprocessing data...")
    train_path, _, _ = ensure_split(processed_dir=processed_dir)
    train_data = pd.read_csv(train_path)

    preprocessor = CustomerChurnPreprocessor()
    train_data_processed = preprocessor.preprocess(train_data)

    X_train = train_data_processed.drop('Exited', axis=1)
    y_train = train_data_processed['Exited']

    logger.info(f"Training Set Size: {X_train.shape}")
    return X_train, y_train


def _cross_val_score(estimator, X_train, y_train):
    pipeline = make_pipeline(
        build_encoder(),
        VarianceThreshold(threshold=0),
        MinMaxScaler(),
        estimator
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

    return scores.mean()


def objective_gb(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_state': 42
    }

    return _cross_val_score(GradientBoostingClassifier(**params), X_train, y_train)


def objective_cat(trial, X_train, y_train):
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

    return _cross_val_score(CatBoostClassifier(**params), X_train, y_train)


def main():
    configure_logging()
    X_train, y_train = load_training_data()

    logger.info("1. Starting Gradient Boosting Optimization...")
    study_gb = optuna.create_study(direction='maximize')
    study_gb.optimize(lambda trial: objective_gb(trial, X_train, y_train), n_trials=N_TRIALS)

    logger.info("\n2. Starting CatBoost Optimization...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(lambda trial: objective_cat(trial, X_train, y_train), n_trials=N_TRIALS)

    logger.info("-" * 50)
    logger.info("RESULTS")
    logger.info("-" * 50)
    logger.info(f"Gradient Boosting Best Score (ROC-AUC): {study_gb.best_value:.4f}")
    logger.info("Best Parameters:", study_gb.best_params)
    logger.info("-" * 50)
    logger.info(f"CatBoost Best Score (ROC-AUC): {study_cat.best_value:.4f}")
    logger.info("Best Parameters:", study_cat.best_params)
    logger.info("-" * 50)


if __name__ == "__main__":
    main()
