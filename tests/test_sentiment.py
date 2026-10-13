import anthropic
import pytest

from src import sentiment
from src.feeds import Article


def art(title, lang, source="s"):
    return Article(source=source, title=title, description="", published=None, link="", language=lang)


def test_label_thresholds():
    assert sentiment.label_for(0.05) == "positive"
    assert sentiment.label_for(-0.05) == "negative"
    assert sentiment.label_for(0.0) == "neutral"


@pytest.mark.parametrize("text,lang", [
    ("The government says the market will recover", "en"),
    ("El gobierno dice que la economía se recupera tras la crisis", "es"),
    ("O governo diz que a economia não vai cair", "pt"),
])
def test_detect_language(text, lang):
    assert sentiment.detect_language(text) == lang


def test_topic_word_boundaries():
    # "said" contains "ai" and "detail" contains "ai": must not be Technology
    assert sentiment.classify_topic("Officials said details were pending", "en") == "General"
    assert sentiment.classify_topic("New AI startup raises funds", "en") == "Technology"
    assert sentiment.classify_topic("Huracán golpea la costa", "es") == "Climate"
    assert sentiment.classify_topic("Inflação e juros pressionam a bolsa", "pt") == "Economy"


def test_auto_offline_routes_by_language():
    rows = sentiment.analyze_batch([art("Floods kill dozens", "en"),
                                    art("Huracán deja muertos", "es"),
                                    art("Enchentes deixam mortos", "pt")])
    assert [r["backend"] for r in rows] == ["vader", "lexicon", "lexicon"]
    assert all(r["sentiment"] == "negative" for r in rows)


def test_missing_language_is_detected():
    rows = sentiment.analyze_batch([art("La economía crece y se recupera el empleo", "")])
    assert rows[0]["language"] == "es" and rows[0]["backend"] == "lexicon"


def test_claude_used_for_non_english_when_client_available(fake_client):
    router = sentiment.BackendRouter(mode="auto", claude_client=fake_client)
    rows = sentiment.analyze_batch([art("Floods kill dozens", "en"), art("Algo pasa", "es")], router=router)
    assert [r["backend"] for r in rows] == ["vader", "claude"]
    assert rows[1]["score"] == 0.5
    call = fake_client.messages.calls[0]
    assert call["model"] == sentiment.DEFAULT_CLAUDE_MODEL
    assert call["output_config"]["format"]["type"] == "json_schema"


def test_claude_model_configurable(fake_client, monkeypatch):
    monkeypatch.setenv("NEWS_SENTIMENT_CLAUDE_MODEL", "some-other-model")
    backend = sentiment.ClaudeBackend(client=fake_client)
    backend.score_batch(["x"], "es")
    assert fake_client.messages.calls[0]["model"] == "some-other-model"


def test_claude_batches_and_clamps(fake_client):
    fake_client.messages.score_fn = lambda text: 7.0  # out of range -> clamped
    backend = sentiment.ClaudeBackend(client=fake_client, batch_size=3)
    scores = backend.score_batch([f"t{i}" for i in range(7)], "pt")
    assert scores == [1.0] * 7
    assert len(fake_client.messages.calls) == 3


def test_claude_forced_for_english(fake_client):
    router = sentiment.BackendRouter(mode="claude", claude_client=fake_client)
    rows = sentiment.analyze_batch([art("Floods kill dozens", "en")], router=router)
    assert rows[0]["backend"] == "claude"


def test_claude_mode_without_key_raises():
    with pytest.raises(RuntimeError):
        sentiment.BackendRouter(mode="claude")


def test_auto_without_key_never_builds_client():
    assert sentiment.BackendRouter(mode="auto").claude is None


@pytest.mark.parametrize("failure", [
    anthropic.APIConnectionError(request=None),
    sentiment.ClaudeBackendError("bad output"),
])
def test_claude_failure_falls_back_offline(fake_client, failure):
    fake_client.messages.fail = failure
    router = sentiment.BackendRouter(mode="auto", claude_client=fake_client)
    rows = sentiment.analyze_batch([art("Huracán deja muertos", "es")], router=router)
    assert rows[0]["backend"] == "lexicon" and rows[0]["sentiment"] == "negative"


def test_claude_refusal_falls_back(fake_client):
    fake_client.messages.stop_reason = "refusal"
    router = sentiment.BackendRouter(mode="auto", claude_client=fake_client)
    rows = sentiment.analyze_batch([art("Enchentes deixam mortos", "pt")], router=router)
    assert rows[0]["backend"] == "lexicon"
