import csv

import main


def test_demo_run_offline_with_charts(tmp_path):
    out = tmp_path / "out"
    assert main.main(["--demo", "--out", str(out)]) == 0
    for name in ["results.csv", "report.md", "01_sentiment_by_source.png", "04_source_topic_heatmap.png"]:
        assert (out / name).exists(), name
    rows = list(csv.DictReader(open(out / "results.csv", encoding="utf-8")))
    assert {r["language"] for r in rows} == {"en", "es", "pt"}
    assert {r["backend"] for r in rows} == {"vader", "lexicon"}


def test_live_mode_with_mocked_network(tmp_path, monkeypatch):
    from tests.test_feeds import RSS
    monkeypatch.setattr(main, "fetch_all",
                        lambda fs, max_per_feed: __import__("src.feeds", fromlist=["x"]).fetch_all(
                            fs, max_per_feed, getter=lambda url: RSS))
    assert main.main(["--languages", "es", "--no-charts", "--out", str(tmp_path)]) == 0
    assert "Huracán" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_claude_backend_requires_key(tmp_path):
    assert main.main(["--demo", "--backend", "claude", "--out", str(tmp_path)]) == 2


def test_missing_input_file_is_a_clean_error(tmp_path, capsys):
    assert main.main(["--input", str(tmp_path / "nope.json"), "--out", str(tmp_path)]) == 2
    assert "Input file not found" in capsys.readouterr().out


def test_invalid_input_json_is_a_clean_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert main.main(["--input", str(bad), "--out", str(tmp_path)]) == 2
    assert "Could not read" in capsys.readouterr().out


def test_csv_cells_are_formula_safe(tmp_path):
    import json
    src = tmp_path / "in.json"
    src.write_text(json.dumps([{"title": "=HYPERLINK(\"http://x\",\"y\")", "language": "en",
                                "source": "@evil", "link": "https://ok.example"}]), encoding="utf-8")
    assert main.main(["--input", str(src), "--no-charts", "--out", str(tmp_path / "o")]) == 0
    row = next(csv.DictReader(open(tmp_path / "o" / "results.csv", encoding="utf-8")))
    assert row["title"].startswith("'=") and row["source"] == "'@evil"


def test_forced_claude_auth_error_exits_nonzero(tmp_path, monkeypatch, capsys):
    import anthropic
    from types import SimpleNamespace
    from src import sentiment
    from tests.conftest import make_status_error

    def fail(**kw):
        raise make_status_error(anthropic.AuthenticationError, 401)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad-key")
    client = SimpleNamespace(messages=SimpleNamespace(create=fail))
    monkeypatch.setattr(main, "BackendRouter",
                        lambda mode: sentiment.BackendRouter(mode=mode, claude_client=client))
    assert main.main(["--demo", "--backend", "claude", "--no-charts", "--out", str(tmp_path)]) == 3
    assert "HTTP 401" in capsys.readouterr().out
