"""Fetches and parses RSS news feeds from public sources."""
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

FEEDS = {
    "Reuters - World":    "https://feeds.reuters.com/reuters/worldNews",
    "Reuters - Business": "https://feeds.reuters.com/reuters/businessNews",
    "BBC - World":        "http://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC - Technology":   "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "AP - Top News":      "https://rsshub.app/apnews/topics/apf-topnews",
}

@dataclass
class Article:
    source: str
    title: str
    description: str
    published: Optional[str]
    link: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.description or ''}"

def fetch_feed(name: str, url: str, max_items: int = 15) -> list[Article]:
    articles = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item") or root.findall(".//atom:entry", ns)
        for item in items[:max_items]:
            title = (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "").strip()
            desc  = (item.findtext("description") or item.findtext("atom:summary", namespaces=ns) or "").strip()
            link  = (item.findtext("link") or item.findtext("atom:link", namespaces=ns) or "").strip()
            pub   = item.findtext("pubDate") or item.findtext("atom:published", namespaces=ns)
            if title:
                articles.append(Article(source=name, title=title,
                                        description=desc, published=pub, link=link))
    except Exception as e:
        print(f"  [warn] {name}: {e}")
    return articles

def fetch_all(max_per_feed: int = 15) -> list[Article]:
    all_articles: list[Article] = []
    for name, url in FEEDS.items():
        articles = fetch_feed(name, url, max_per_feed)
        print(f"  {name}: {len(articles)} articles")
        all_articles.extend(articles)
    return all_articles
