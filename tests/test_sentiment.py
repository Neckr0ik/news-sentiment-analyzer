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


def test_topic_fixes():
    assert sentiment.classify_topic("Floods hit the coast", "en") == "Climate"
    assert sentiment.classify_topic("América gana el partido ante Chivas", "es") == "General"
    assert sentiment.classify_topic("Nuevo partido político pide registro", "es") == "Politics"


def test_topic_tie_goes_to_first_listed():
    # one Politics keyword and one Economy keyword: Politics is listed first
    assert sentiment.classify_topic("Government weighs inflation", "en") == "Politics"


from tests.conftest import make_status_error as _status_error


def test_fallback_is_per_chunk(fake_client):
    calls = {"n": 0}
    real_create = fake_client.messages.create

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise anthropic.APIConnectionError(request=None)
        return real_create(**kwargs)

    fake_client.messages.create = flaky
    router = sentiment.BackendRouter(mode="auto", claude_client=fake_client)
    router.claude.batch_size = 2
    rows = sentiment.analyze_batch([art(f"Huracán deja muertos {i}", "es") for i in range(5)], router=router)
    assert [r["backend"] for r in rows] == ["claude", "claude", "lexicon", "lexicon", "claude"]


def test_malformed_json_falls_back(fake_client):
    from types import SimpleNamespace
    fake_client.messages.create = lambda **kw: SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text="{not json")])
    router = sentiment.BackendRouter(mode="auto", claude_client=fake_client)
    rows = sentiment.analyze_batch([art("Huracán deja muertos", "es")], router=router)
    assert rows[0]["backend"] == "lexicon"


def test_programming_errors_are_not_swallowed(fake_client):
    fake_client.messages.fail = ZeroDivisionError("bug")
    router = sentiment.BackendRouter(mode="auto", claude_client=fake_client)
    with pytest.raises(ZeroDivisionError):
        sentiment.analyze_batch([art("Huracán deja muertos", "es")], router=router)


@pytest.mark.parametrize("cls,code", [
    (anthropic.BadRequestError, 400), (anthropic.AuthenticationError, 401),
    (anthropic.PermissionDeniedError, 403), (anthropic.NotFoundError, 404),
])
def test_forced_claude_stops_on_request_errors(fake_client, cls, code):
    fake_client.messages.fail = _status_error(cls, code)
    router = sentiment.BackendRouter(mode="claude", claude_client=fake_client)
    with pytest.raises(sentiment.ClaudeConfigError, match=str(code)):
        sentiment.analyze_batch([art("Floods kill dozens", "en")], router=router)


def test_auto_disables_claude_after_auth_error(fake_client):
    fake_client.messages.fail = _status_error(anthropic.AuthenticationError, 401)
    router = sentiment.BackendRouter(mode="auto", claude_client=fake_client)
    router.claude.batch_size = 1
    rows = sentiment.analyze_batch([art("Huracán deja muertos", "es"), art("Enchentes deixam mortos", "pt")],
                                   router=router)
    assert [r["backend"] for r in rows] == ["lexicon", "lexicon"]
    assert len(fake_client.messages.calls) == 1   # no further calls after the 401


def test_server_error_falls_back_in_forced_mode(fake_client):
    fake_client.messages.fail = _status_error(anthropic.InternalServerError, 500)
    router = sentiment.BackendRouter(mode="claude", claude_client=fake_client)
    rows = sentiment.analyze_batch([art("Floods kill dozens", "en")], router=router)
    assert rows[0]["backend"] == "vader"
