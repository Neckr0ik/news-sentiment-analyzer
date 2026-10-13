import pandas as pd

from src import reporter


def _df(title, link, source="s"):
    return pd.DataFrame([{"source": source, "language": "es", "title": title, "topic": "General",
                          "sentiment": "negative", "score": -0.5, "backend": "lexicon", "link": link}])


def _report(tmp_path, df):
    path = tmp_path / "r.md"
    reporter.write_markdown_report(df, str(path))
    return path.read_text(encoding="utf-8")


def test_title_markup_is_escaped(tmp_path):
    text = _report(tmp_path, _df("Click [here](https://evil.example) <img src=x>", "https://ok.example/a"))
    assert "[here](https://evil.example)" not in text
    assert "<img" not in text
    assert "\\[here\\]\\(https://evil.example\\)" in text
    assert "](https://ok.example/a)" in text


def test_only_http_links_are_kept(tmp_path):
    text = _report(tmp_path, _df("Titular", "javascript:alert(1)"))
    assert "javascript" not in text
    assert "- Titular `-0.500`" in text


def test_empty_link_has_no_empty_brackets(tmp_path):
    text = _report(tmp_path, _df("Titular", ""))
    assert "]()" not in text and "[Titular]" not in text


def test_link_cannot_break_out_of_target(tmp_path):
    text = _report(tmp_path, _df("T", "https://ok.example/a) [x](https://evil"))
    assert ") [x](" not in text
    assert "https://ok.example/a%29%20" in text


def test_csv_safe_neutralises_formulas():
    df = pd.DataFrame({"title": ["=HYPERLINK(\"x\")", "+1", "-2", "@SUM(A1)", "ok"], "score": [-0.5, 0.1, 0, 1, -1]})
    out = reporter.csv_safe(df)
    assert list(out["title"]) == ["'=HYPERLINK(\"x\")", "'+1", "'-2", "'@SUM(A1)", "ok"]
    assert list(out["score"]) == [-0.5, 0.1, 0, 1, -1]
    assert df["title"][0].startswith("=")  # original untouched
