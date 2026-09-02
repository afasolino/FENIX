from qualification.runtime_lane import (
    ENVIRONMENT_BLOCKED,
    READY_FOR_MODEL_FETCH,
    RUNTIME_QUALIFIED,
)


def test_model_fetch_status_vocabulary_is_unambiguous():
    assert ENVIRONMENT_BLOCKED != READY_FOR_MODEL_FETCH
    assert RUNTIME_QUALIFIED != READY_FOR_MODEL_FETCH
    assert RUNTIME_QUALIFIED != ENVIRONMENT_BLOCKED
