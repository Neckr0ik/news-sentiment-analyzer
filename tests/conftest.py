import json
from types import SimpleNamespace

import pytest


class FakeMessages:
    """Stands in for client.messages; returns canned structured output, records calls."""

    def __init__(self, score_fn=lambda text: 0.5, stop_reason="end_turn", fail=None):
        self.calls = []
        self.score_fn = score_fn
        self.stop_reason = stop_reason
        self.fail = fail

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        content = kwargs["messages"][0]["content"]
        items = json.loads(content.split("Items:\n", 1)[1])
        payload = {"results": [{"id": it["id"], "score": self.score_fn(it["text"])} for it in items]}
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="text", text=json.dumps(payload))])


@pytest.fixture
def fake_client():
    return SimpleNamespace(messages=FakeMessages())


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    # Tests must never reach the network: make sure no real key is picked up.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
