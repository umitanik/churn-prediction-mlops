import logging
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

RAW_PATH = os.path.join('data', 'raw', 'Customer-Churn-Records.csv')
PROCESSED_DIR = os.path.join('data', 'processed')

TARGET_COL = 'Exited'
TEST_SIZE = 0.1
RANDOM_STATE = 42


def split_paths(processed_dir=PROCESSED_DIR):
    return (
        os.path.join(processed_dir, 'train.csv'),
        os.path.join(processed_dir, 'test.csv'),
        os.path.join(processed_dir, 'ground_truth.csv'),
    )


def split_exists(processed_dir=PROCESSED_DIR):
    return all(os.path.exists(path) for path in split_paths(processed_dir))


def create_split(raw_path=RAW_PATH, processed_dir=PROCESSED_DIR):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path}. "
            "It is tracked in the repository, so make sure the working tree is complete."
        )

    data = pd.read_csv(raw_path)
    logger.info(f"Raw dataset loaded: {data.shape}")

    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=data[TARGET_COL],
    )

    os.makedirs(processed_dir, exist_ok=True)
    train_path, test_path, ground_truth_path = split_paths(processed_dir)

    train_data.to_csv(train_path, index=False)
    test_data.drop(columns=TARGET_COL).to_csv(test_path, index=False)
    test_data[[TARGET_COL]].to_csv(ground_truth_path, index=False)

    logger.info(f"Train set saved   ({train_data.shape}): {train_path}")
    logger.info(f"Test set saved    ({test_data.shape[0]} rows, target removed): {test_path}")
    logger.info(f"Ground truth saved ({test_data.shape[0]} rows): {ground_truth_path}")

    return train_path, test_path, ground_truth_path


def ensure_split(raw_path=RAW_PATH, processed_dir=PROCESSED_DIR, force=False):
    if split_exists(processed_dir) and not force:
        logger.info(f"Processed split already present in {processed_dir}, skipping.")
        return split_paths(processed_dir)

    return create_split(raw_path, processed_dir)


def main():
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Create the reproducible train/test split from the raw dataset."
    )
    parser.add_argument('--raw-path', default=RAW_PATH, help="Path to the raw CSV file.")
    parser.add_argument('--processed-dir', default=PROCESSED_DIR, help="Output directory for the split.")
    parser.add_argument('--force', action='store_true', help="Recreate the split even if it already exists.")
    args = parser.parse_args()

    ensure_split(args.raw_path, args.processed_dir, force=args.force)


if __name__ == "__main__":
    main()
