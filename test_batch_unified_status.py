import pytest
from types import SimpleNamespace
from ia_provider.batch import BatchJobManager


@pytest.mark.parametrize("raw,expected", [
    ("in_progress", "running"),
    ("completed", "completed"),
    ("failed", "failed"),
    ("cancelled", "failed"),
])
def test_unify_status_openai_mapping(raw, expected):
    manager = BatchJobManager(api_key="", provider_type="openai")
    result = manager._unify_status({"status": raw, "provider": "openai"})
    assert result["unified_status"] == expected
    assert result["status"] == raw


@pytest.mark.parametrize("raw,expected", [
    ("processing", "running"),
    ("ended", "completed"),
    ("expired", "failed"),
])
def test_unify_status_anthropic_mapping(raw, expected):
    manager = BatchJobManager(api_key="", provider_type="anthropic")
    result = manager._unify_status({"status": raw, "provider": "anthropic"})
    assert result["unified_status"] == expected
    assert result["status"] == raw


def test_get_status_and_history_include_unified_status_openai():
    manager = BatchJobManager(api_key="", provider_type="openai")

    batch_status = SimpleNamespace(
        id="batch_b1",
        status="in_progress",
        created_at=0,
        endpoint="/v1/chat",
        completion_window="24h",
        request_counts=None,
        output_file_id=None,
        error_file_id=None,
        input_file_id=None,
        metadata={},
    )

    batch_history = SimpleNamespace(
        id="b2",
        status="completed",
        created_at=0,
        endpoint="/v1/chat",
        completion_window="24h",
        request_counts=None,
        output_file_id=None,
        error_file_id=None,
        metadata={},
    )

    manager.client = SimpleNamespace(
        batches=SimpleNamespace(
            retrieve=lambda batch_id: batch_status,
            list=lambda limit: SimpleNamespace(data=[batch_history]),
        )
    )

    status = manager.get_status("batch_b1")
    assert status["status"] == "in_progress"
    assert status["unified_status"] == "running"

    history = manager.get_history(limit=1)
    assert len(history) == 1
    assert history[0]["status"] == "completed"
    assert history[0]["unified_status"] == "completed"
    assert "unified_status" in status and "unified_status" in history[0]
