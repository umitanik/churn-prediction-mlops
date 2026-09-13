import logging
import argparse
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

from src.data.split_dataset import PROCESSED_DIR, ensure_split
from src.ml.preprocessor import CustomerChurnPreprocessor, build_encoder
from src.ml.threshold import choose_threshold

MODEL_DIR = os.getenv('MODEL_DIR', 'models')
MODEL_FILENAME = 'catboost_churn_model.pkl'
RECALL_WEIGHT = 2.0   # beta for F-beta: a missed churner costs more than a wasted call

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

def build_pipeline(params=None):
    return make_pipeline(
        build_encoder(),
        VarianceThreshold(threshold=0),
        MinMaxScaler(),
        CatBoostClassifier(**(params or best_params_catboost))
    )
    return make_pipeline(
        encoder,
        VarianceThreshold(threshold=0),
        MinMaxScaler(),
        CatBoostClassifier(**(params or best_params_catboost))
    )

def load_datasets(processed_dir=PROCESSED_DIR):
    train_path, test_path, ground_truth_path = ensure_split(processed_dir=processed_dir)

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    ground_truth = pd.read_csv(ground_truth_path)

    preprocessor = CustomerChurnPreprocessor()
    train_data_processed = preprocessor.preprocess(train_data)
    test_data_processed = preprocessor.preprocess(test_data)

    X_train = train_data_processed.drop('Exited', axis=1)
    y_train = train_data_processed['Exited']

    X_test = test_data_processed
    y_test = ground_truth['Exited']

    return X_train, y_train, X_test, y_test


def train(model_dir=MODEL_DIR, processed_dir=PROCESSED_DIR, recall_weight=RECALL_WEIGHT):
    logger.info("Starting Final Model Training (CatBoost)...")

    logger.info("Loading data...")
    X_train, y_train, X_test, y_test = load_datasets(processed_dir)

    final_pipeline = build_pipeline()
    final_pipeline.fit(X_train, y_train)

    final_pipeline.model_version_ = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    logger.info("Choosing the decision threshold on out-of-fold predictions...")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = cross_val_predict(build_pipeline(), X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    threshold, oof_fbeta = choose_threshold(y_train, oof_prob, beta=recall_weight)
    final_pipeline.decision_threshold_ = threshold
    logger.info(f" Decision threshold: {threshold:.4f}  (F{recall_weight:g} on OOF = {oof_fbeta:.4f})")

    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, y_prob)

    logger.info("\n" + "="*40)
    logger.info(f" FINAL ROC-AUC SCORE: {roc_score:.4f}")
    logger.info("="*40)

    logger.info("Test set at the default 0.5 threshold:\n%s",
                classification_report(y_test, (y_prob >= 0.5).astype(int)))
    logger.info("Test set at the chosen threshold %.4f:\n%s", threshold,
                classification_report(y_test, (y_prob >= threshold).astype(int)))

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    joblib.dump(final_pipeline, model_path)
    logger.info(f"\n Model and Pipeline successfully saved to: {model_path}")
    logger.info(f" Model version: {final_pipeline.model_version_}")
    logger.info(f" Decision threshold: {final_pipeline.decision_threshold_:.4f}")

    return model_path, roc_score


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Train the final CatBoost churn model.")
    parser.add_argument('--model-dir', default=MODEL_DIR, help="Directory the model artifact is written to.")
    parser.add_argument('--processed-dir', default=PROCESSED_DIR, help="Directory holding the train/test split.")
    parser.add_argument(
        '--recall-weight', type=float, default=RECALL_WEIGHT,
        help="beta for the F-beta threshold objective; >1 favours recall over precision "
             f"(default {RECALL_WEIGHT:g}). Replace with a cost model once real numbers exist."
    )
    parser.add_argument(
        '--skip-if-exists',
        action='store_true',
        help="Exit without training when the model artifact is already present."
    )
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, MODEL_FILENAME)
    if args.skip_if_exists and os.path.exists(model_path):
        logger.info(f"Model artifact already present, skipping training: {model_path}")
        return

    train(args.model_dir, args.processed_dir, args.recall_weight)


if __name__ == "__main__":
    main()
