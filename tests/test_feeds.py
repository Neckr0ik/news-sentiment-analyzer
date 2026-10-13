import json

from src import feeds

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>X</title>
<item><title><![CDATA[Hurac\xc3\xa1n deja da\xc3\xb1os]]></title>
<link>https://example.mx/a</link>
<description><![CDATA[<p>Texto con <b>HTML</b> &amp; entidades</p>]]></description>
<pubDate>Tue, 13 Oct 2026 04:03:42 +0000</pubDate></item>
<item><title></title><link>https://example.mx/empty</link></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Y</title>
<entry><title>Markets rally</title><link href="https://example.com/b"/>
<summary>Stocks rose.</summary><updated>2026-10-13T00:00:00Z</updated></entry>
</feed>"""


def test_parse_rss_cleans_html_and_skips_empty_titles():
    arts = feeds.parse_feed(RSS, "Ejemplo", "es")
    assert len(arts) == 1
    a = arts[0]
    assert a.title == "Huracán deja daños"
    assert a.description == "Texto con HTML & entidades"
    assert a.link == "https://example.mx/a" and a.language == "es"


def test_parse_atom_reads_href_links():
    arts = feeds.parse_feed(ATOM, "Atom", "en")
    assert arts[0].title == "Markets rally"
    assert arts[0].link == "https://example.com/b"
    assert arts[0].published == "2026-10-13T00:00:00Z"


def test_fetch_all_uses_getter_and_survives_errors():
    def getter(url):
        if "bad" in url:
            raise OSError("network down")
        return RSS
    fs = [feeds.Feed("ok", "https://ok"), feeds.Feed("bad", "https://bad")]
    arts = feeds.fetch_all(fs, getter=getter)
    assert len(arts) == 1 and arts[0].source == "ok"


def test_default_config_has_spanish_mexican_feeds():
    es = feeds.load_feeds(languages=["es"])
    assert len(es) >= 4
    assert any(f.region == "MX" for f in es)
    assert {f.language for f in feeds.load_feeds()} == {"en", "es", "pt"}


def test_load_articles_json(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([{"title": "Hola", "language": "es"}]), encoding="utf-8")
    arts = feeds.load_articles_json(str(p))
    assert arts[0].title == "Hola" and arts[0].source == "local"


def test_entity_declarations_are_refused():
    bomb = b"""<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;">]>
<rss><channel><item><title>&b;</title></item></channel></rss>"""
    assert feeds.fetch_all([feeds.Feed("bomb", "https://x")], getter=lambda url: bomb) == []


def test_double_encoded_html_is_stripped():
    assert feeds.clean_text("&amp;lt;script&amp;gt;alert(1)&amp;lt;/script&amp;gt; Hola") == "alert(1) Hola"


def test_http_get_caps_response_size(monkeypatch):
    import io
    import pytest

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(feeds, "MAX_FEED_BYTES", 10)
    monkeypatch.setattr(feeds.urllib.request, "urlopen", lambda req, timeout: Resp(b"x" * 50))
    with pytest.raises(ValueError, match="larger than"):
        feeds._http_get("https://example.com/feed")
