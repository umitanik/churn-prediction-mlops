import pytest

from src.ml.preprocessor import CAT_COLS
from tests.conftest import score, unvalidated_frame

EXPECTED_INPUT_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Satisfaction Score", "Card Type", "Point Earned",
    "BalanceSalaryRatio", "TenureByAge", "HasBalance", "CreditScoreGivenAge",
]


def test_scoring_one_row_matches_scoring_it_in_a_batch(
    trained_model, preprocessor, train_frame
):
    """The single most important assertion in this suite.

    Serving passes one row; training passes nine thousand. Any step whose
    behaviour depends on how many rows it sees will make these two disagree,
    which is exactly how the original defect went unnoticed.
    """
    row = train_frame.iloc[[0]].copy()

    alone = trained_model.predict_proba(preprocessor.preprocess(row))[0][1]
    in_batch = trained_model.predict_proba(
        preprocessor.preprocess(train_frame.copy())
    )[0][1]

    assert alone == pytest.approx(in_batch, abs=1e-12)


def test_pipeline_consumes_raw_columns(trained_model):
    """The encoder lives inside the artifact, so it expects unencoded input.

    If this list ever contains names like ``Geography_Germany`` again, encoding
    has leaked back out of the pipeline and the serving layer is once more
    responsible for reproducing it.
    """
    assert list(trained_model.feature_names_in_) == EXPECTED_INPUT_COLUMNS


def test_encoder_carries_the_training_categories(trained_model):
    """The categories must be stored in the artifact, not derived per request."""
    encoder = trained_model.named_steps["columntransformer"]
    learned = dict(zip(CAT_COLS, encoder.named_transformers_["cat"].categories_))

    assert list(learned["Geography"]) == ["France", "Germany", "Spain"]
    assert list(learned["Gender"]) == ["Female", "Male"]
    assert list(learned["Card Type"]) == ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]


@pytest.mark.parametrize(
    "field, values",
    [
        ("Geography", ["France", "Germany", "Spain"]),
        ("Gender", ["Female", "Male"]),
        ("CardType", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]),
    ],
)
def test_categorical_values_change_the_prediction(
    trained_model, preprocessor, payload, field, values
):
    """Parity alone is not enough.

    A feature that is ignored entirely would still satisfy the parity test,
    because both paths would agree on the same wrong answer. This asserts the
    categorical inputs actually reach the model and move the output.
    """
    probabilities = {
        value: score(trained_model, preprocessor, payload, **{field: value})
        for value in values
    }

    assert len(set(probabilities.values())) > 1, (
        f"{field} does not influence the prediction: {probabilities}"
    )


def test_unknown_category_is_rejected(trained_model, preprocessor, payload):
    """`handle_unknown='error'` must stay.

    Switching it to 'ignore' would make an unseen value encode as all-zeros,
    which is indistinguishable from the dropped reference category. The request
    schema normally rejects this input first (see test_api.py); here it is
    bypassed deliberately so the encoder's own defence is exercised.
    """
    frame = unvalidated_frame(payload, Geography="Atlantis")

    with pytest.raises(ValueError, match="unknown categories"):
        trained_model.predict_proba(preprocessor.preprocess(frame))


def test_missing_feature_is_rejected(trained_model, preprocessor, payload):
    """No silent padding.

    The serving code used to call reindex(..., fill_value=0), which turned a
    column mismatch into a confident wrong answer instead of an error.
    """
    features = preprocessor.preprocess(unvalidated_frame(payload)).drop(
        columns=["Point Earned"]
    )

    with pytest.raises(ValueError, match="columns are missing"):
        trained_model.predict_proba(features)


def test_probability_is_a_valid_probability(trained_model, preprocessor, payload):
    probability = score(trained_model, preprocessor, payload)

    assert 0.0 <= probability <= 1.0
