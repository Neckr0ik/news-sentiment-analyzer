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
