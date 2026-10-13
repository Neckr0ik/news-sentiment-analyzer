"""Fetches and parses RSS 2.0 / Atom news feeds listed in config/feeds.json."""
from __future__ import annotations

import html
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

DEFAULT_FEEDS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "feeds.json")
ATOM = "{http://www.w3.org/2005/Atom}"
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Feed:
    name: str
    url: str
    language: str = "en"
    region: str = ""


@dataclass
class Article:
    source: str
    title: str
    description: str
    published: Optional[str]
    link: str
    language: str = "en"

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.description or ''}".strip()


def load_feeds(path: str = DEFAULT_FEEDS_PATH,
               languages: Optional[Iterable[str]] = None) -> list[Feed]:
    """Load feed definitions, optionally keeping only the given languages."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    wanted = {lang.strip().lower() for lang in languages} if languages else None
    feeds = [Feed(**{k: v for k, v in entry.items() if k in Feed.__dataclass_fields__})
             for entry in data["feeds"]]
    if wanted:
        feeds = [f for f in feeds if f.language.lower() in wanted]
    return feeds


def clean_text(raw: str) -> str:
    """Strip HTML tags/entities that many feeds embed in <description>."""
    text = html.unescape(_TAG_RE.sub(" ", raw or ""))
    return re.sub(r"\s+", " ", text).strip()


def parse_feed(raw: bytes, name: str, language: str = "en", max_items: int = 15) -> list[Article]:
    """Parse RSS 2.0 or Atom XML into Article objects (no network access)."""
    root = ET.fromstring(raw)
    articles: list[Article] = []

    items = root.findall(".//item")
    if items:  # RSS 2.0
        for item in items[:max_items]:
            title = clean_text(item.findtext("title") or "")
            if not title:
                continue
            articles.append(Article(
                source=name,
                title=title,
                description=clean_text(item.findtext("description") or ""),
                published=item.findtext("pubDate"),
                link=(item.findtext("link") or "").strip(),
                language=language,
            ))
        return articles

    for entry in root.findall(f".//{ATOM}entry")[:max_items]:  # Atom
        title = clean_text(entry.findtext(f"{ATOM}title") or "")
        if not title:
            continue
        link_el = entry.find(f"{ATOM}link")
        link = link_el.get("href", "") if link_el is not None else ""
        articles.append(Article(
            source=name,
            title=title,
            description=clean_text(entry.findtext(f"{ATOM}summary") or ""),
            published=entry.findtext(f"{ATOM}published") or entry.findtext(f"{ATOM}updated"),
            link=link,
            language=language,
        ))
    return articles


def _http_get(url: str, timeout: float = 8.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "news-sentiment-analyzer/1.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_feed(feed: Feed, max_items: int = 15,
               getter: Callable[[str], bytes] = _http_get) -> list[Article]:
    try:
        return parse_feed(getter(feed.url), feed.name, feed.language, max_items)
    except Exception as e:  # network errors, bad XML: skip the feed, keep the run going
        print(f"  [warn] {feed.name}: {e}")
        return []


def fetch_all(feeds: list[Feed], max_per_feed: int = 15,
              getter: Callable[[str], bytes] = _http_get) -> list[Article]:
    all_articles: list[Article] = []
    for feed in feeds:
        articles = fetch_feed(feed, max_per_feed, getter)
        print(f"  {feed.name} [{feed.language}]: {len(articles)} articles")
        all_articles.extend(articles)
    return all_articles


def load_articles_json(path: str) -> list[Article]:
    """Load articles from a local JSON list (used by --input / --demo, fully offline)."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [Article(source=r.get("source", "local"), title=r["title"],
                    description=r.get("description", ""), published=r.get("published"),
                    link=r.get("link", ""), language=r.get("language", "en"))
            for r in rows]
