import pandas as pd
import pytest

from src.ml.preprocessor import CAT_COLS, CustomerChurnPreprocessor

ENGINEERED_FEATURES = [
    "BalanceSalaryRatio",
    "TenureByAge",
    "HasBalance",
    "CreditScoreGivenAge",
]


def test_single_row_and_batch_produce_the_same_columns(preprocessor, train_frame):
    """The invariant that broke: one row must be shaped like the whole batch.

    The original implementation called pd.get_dummies here, which derives its
    output columns from the frame it is handed. On a single row every
    categorical column holds one value, drop_first removed the only dummy, and
    the categorical columns vanished entirely.
    """
    single = preprocessor.preprocess(train_frame.iloc[[0]].copy())
    batch = preprocessor.preprocess(train_frame.copy())

    assert list(single.columns) == list(batch.columns)


def test_single_row_and_batch_produce_the_same_values(preprocessor, train_frame):
    single = preprocessor.preprocess(train_frame.iloc[[0]].copy())
    batch = preprocessor.preprocess(train_frame.copy())

    pd.testing.assert_frame_equal(
        single.reset_index(drop=True),
        batch.iloc[[0]].reset_index(drop=True),
    )


def test_categorical_columns_are_left_unencoded(preprocessor, train_frame):
    """Encoding is the pipeline's job, so the raw categories must survive.

    Checked on the values rather than the dtype: pandas 2 stores text as
    object, pandas 3 as StringDtype, and either is fine as long as no one-hot
    columns have appeared and the original labels are still there.
    """
    out = preprocessor.preprocess(train_frame.copy())

    for column in CAT_COLS:
        assert column in out.columns
        assert pd.api.types.is_string_dtype(out[column])
        assert set(out[column].unique()) == set(train_frame[column].unique())
        assert not any(c.startswith(f"{column}_") for c in out.columns)


@pytest.mark.parametrize("column", ["RowNumber", "CustomerId", "Surname"])
def test_identifier_columns_are_dropped(preprocessor, train_frame, column):
    """Identifiers carry no signal and must not reach the model."""
    frame = train_frame.copy()
    if column not in frame.columns:
        frame[column] = 1

    assert column not in preprocessor.preprocess(frame).columns


def test_complain_is_dropped_to_prevent_target_leakage(preprocessor, train_frame):
    """`Complain` correlates with the target at 0.996.

    Leaving it in produces a model that looks near-perfect in evaluation and is
    useless in production, because the complaint is registered as part of the
    churn event itself. This assertion is the guard against it silently
    returning.
    """
    frame = train_frame.copy()
    frame["Complain"] = 0

    assert "Complain" not in preprocessor.preprocess(frame).columns


def test_engineered_features_are_added(preprocessor, train_frame):
    out = preprocessor.preprocess(train_frame.copy())

    for feature in ENGINEERED_FEATURES:
        assert feature in out.columns


def test_engineered_features_use_the_documented_formulas(preprocessor):
    """Pin the arithmetic so a typo in a ratio cannot pass unnoticed."""
    row = pd.DataFrame([{
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 4,
        "Balance": 50_000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 100_000.0,
        "Satisfaction Score": 3,
        "Card Type": "GOLD",
        "Point Earned": 400,
    }])

    out = preprocessor.preprocess(row).iloc[0]

    assert out["BalanceSalaryRatio"] == pytest.approx(0.5, abs=1e-6)
    assert out["TenureByAge"] == pytest.approx(0.1, abs=1e-6)
    assert out["CreditScoreGivenAge"] == pytest.approx(15.0, abs=1e-6)
    assert out["HasBalance"] == 1


def test_has_balance_flags_a_zero_balance(preprocessor, train_frame):
    frame = train_frame.copy()
    frame.loc[frame.index[0], "Balance"] = 0.0

    assert preprocessor.preprocess(frame).iloc[0]["HasBalance"] == 0


def test_preprocess_does_not_mutate_its_input(preprocessor, train_frame):
    """Callers hand over frames they still need; preprocess must copy."""
    frame = train_frame.head(5).copy()
    before = frame.copy()

    preprocessor.preprocess(frame)

    pd.testing.assert_frame_equal(frame, before)
