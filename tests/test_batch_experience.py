import json
from types import SimpleNamespace

import pytest

import ia_provider.batch as batch_module
from ia_provider.batch import BatchJobManager


@pytest.mark.parametrize(
    "provider,raw,expected",
    [
        ("openai", "in_progress", "running"),
        ("openai", "completed", "completed"),
        ("openai", "failed", "failed"),
        ("anthropic", "processing", "running"),
        ("anthropic", "ended", "completed"),
        ("anthropic", "expired", "failed"),
    ],
)
def test_unify_status(provider, raw, expected):
    manager = BatchJobManager(api_key="", provider_type=provider)
    result = manager._unify_status({"status": raw, "provider": provider})
    assert result["unified_status"] == expected


def test_clean_response_openai():
    manager = BatchJobManager(api_key="", provider_type="openai")

    batch = SimpleNamespace(status="completed", output_file_id="out", error_file_id=None)

    line = json.dumps(
        {
            "custom_id": "1",
            "response": {
                "body": {
                    "choices": [
                        {"message": {"content": "hello world"}}
                    ]
                }
            },
        }
    )

    manager.client = SimpleNamespace(
        batches=SimpleNamespace(retrieve=lambda _id: batch),
        files=SimpleNamespace(content=lambda _id: SimpleNamespace(text=line)),
    )

    results = manager.get_results("batch_1")
    assert results[0].clean_response == "hello world"


def test_clean_response_anthropic():
    manager = BatchJobManager(api_key="", provider_type="anthropic")

    message = SimpleNamespace(content=[SimpleNamespace(text="hi there")], role="assistant")
    result_obj = SimpleNamespace(
        custom_id="1",
        result=SimpleNamespace(type="succeeded", message=message),
        model_dump=lambda: {},
    )
    batch = SimpleNamespace(processing_status="ended")

    manager.client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                batches=SimpleNamespace(
                    retrieve=lambda _id: batch,
                    results=lambda _id: [result_obj],
                )
            )
        )
    )

    results = manager.get_results("batch_1")
    assert results[0].clean_response == "hi there"


def test_local_history_save_and_load(tmp_path, monkeypatch):
    history_file = tmp_path / "batch_history.json"
    monkeypatch.setattr(batch_module, "HISTORY_FILE", history_file)

    batch_module._save_batch_to_local_history("b1", "openai")
    data = batch_module._load_local_batch_history()
    assert data[0]["id"] == "b1"
    assert data[0]["provider"] == "openai"

    # saving same id should not create duplicates
    batch_module._save_batch_to_local_history("b1", "openai")
    data = batch_module._load_local_batch_history()
    assert len(data) == 1

