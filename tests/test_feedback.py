import pytest


def _predict(client, payload):
    return client.post("/predict", json=payload).json()["log_id"]


def _log(client, log_id):
    logs = client.get("/logs", params={"limit": 100}).json()
    return next(entry for entry in logs if entry["id"] == log_id)


def test_new_prediction_has_no_outcome_yet(client, payload):
    entry = _log(client, _predict(client, payload))

    assert entry["actual_label"] is None
    assert entry["labeled_at"] is None


def test_outcome_is_recorded_with_a_timestamp(client, payload):
    log_id = _predict(client, payload)

    response = client.post(f"/feedback/{log_id}", json={"actual_label": "CHURN"})

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == log_id
    assert body["actual_label"] == "CHURN"
    assert body["labeled_at"] is not None


def test_outcome_is_visible_in_the_log_afterwards(client, payload):
    log_id = _predict(client, payload)
    client.post(f"/feedback/{log_id}", json={"actual_label": "LOYAL"})

    assert _log(client, log_id)["actual_label"] == "LOYAL"


def test_recording_the_same_outcome_twice_is_a_no_op(client, payload):
    log_id = _predict(client, payload)
    first = client.post(f"/feedback/{log_id}", json={"actual_label": "CHURN"}).json()

    second = client.post(f"/feedback/{log_id}", json={"actual_label": "CHURN"})

    assert second.status_code == 200
    assert second.json()["labeled_at"] == first["labeled_at"]


def test_an_outcome_cannot_be_changed_once_written(client, payload):
    """Rewriting ground truth would let the training set be edited after the fact."""
    log_id = _predict(client, payload)
    client.post(f"/feedback/{log_id}", json={"actual_label": "CHURN"})

    response = client.post(f"/feedback/{log_id}", json={"actual_label": "LOYAL"})

    assert response.status_code == 409
    assert _log(client, log_id)["actual_label"] == "CHURN"


def test_unknown_prediction_is_not_found(client):
    assert client.post("/feedback/999999", json={"actual_label": "CHURN"}).status_code == 404


@pytest.mark.parametrize("label", ["churn", "YES", "1", ""])
def test_only_the_two_known_labels_are_accepted(client, payload, label):
    log_id = _predict(client, payload)

    assert client.post(f"/feedback/{log_id}", json={"actual_label": label}).status_code == 422


def test_feedback_requires_the_api_key(client, payload):
    log_id = _predict(client, payload)

    response = client.post(
        f"/feedback/{log_id}", json={"actual_label": "CHURN"}, headers={"X-API-Key": "wrong"}
    )

    assert response.status_code == 401
    assert _log(client, log_id)["actual_label"] is None
