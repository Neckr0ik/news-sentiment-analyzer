"""
News Sentiment Analyzer

    python main.py                         # live feeds (en/es/pt), charts + report in output/
    python main.py --languages es          # only Spanish-language feeds
    python main.py --demo --no-charts      # offline run on bundled sample headlines
    python main.py --backend claude        # force the Claude backend (needs ANTHROPIC_API_KEY)

English is scored with VADER. Spanish/Portuguese use the Claude API when
ANTHROPIC_API_KEY is set, otherwise an offline curated lexicon.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from src.feeds import DEFAULT_FEEDS_PATH, fetch_all, load_articles_json, load_feeds
from src.sentiment import BackendRouter, analyze_batch, claude_available

DEMO_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_headlines.json")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score the tone of news headlines (en/es/pt).")
    p.add_argument("--languages", default="en,es,pt",
                   help="comma-separated feed languages to fetch (default: en,es,pt)")
    p.add_argument("--backend", choices=["auto", "vader", "lexicon", "claude"], default="auto",
                   help="sentiment backend (default: auto)")
    p.add_argument("--feeds", default=DEFAULT_FEEDS_PATH, help="path to a feeds JSON config")
    p.add_argument("--max-per-feed", type=int, default=15)
    p.add_argument("--input", help="score a local JSON list of articles instead of fetching feeds")
    p.add_argument("--demo", action="store_true", help="offline run on data/sample_headlines.json")
    p.add_argument("--out", default="output", help="output directory (default: output)")
    p.add_argument("--no-charts", action="store_true", help="skip chart generation")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print("News Sentiment Analyzer")
    print("=" * 50)

    if args.backend == "claude" and not claude_available():
        print("  --backend claude requires ANTHROPIC_API_KEY to be set.")
        return 2

    print("\n[1/3] Loading articles...")
    if args.demo or args.input:
        path = args.input or DEMO_PATH
        articles = load_articles_json(path)
        print(f"  Loaded {len(articles)} articles from {path}")
    else:
        feeds = load_feeds(args.feeds, args.languages.split(","))
        articles = fetch_all(feeds, max_per_feed=args.max_per_feed)
        print(f"  Total articles fetched: {len(articles)}")

    if not articles:
        print("  No articles to analyze. Check your internet connection or try --demo.")
        return 1

    print("\n[2/3] Running sentiment analysis...")
    if args.backend == "auto":
        print("  Spanish/Portuguese backend: " + ("Claude API" if claude_available() else "offline lexicon"))
    df = pd.DataFrame(analyze_batch(articles, router=BackendRouter(mode=args.backend)))

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False, encoding="utf-8")
    print(f"  Analyzed {len(df)} articles across {df['source'].nunique()} sources")

    print("\n[3/3] Writing report" + ("" if args.no_charts else " and charts") + "...")
    if not args.no_charts:
        from src.reporter import plot_all  # matplotlib import only when needed
        for path in plot_all(df, args.out):
            print(f"  chart  -> {path}")
    from src.reporter import write_markdown_report
    print(f"  report -> {write_markdown_report(df, os.path.join(args.out, 'report.md'))}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Articles:  {len(df)}")
    for label, count in df["sentiment"].value_counts().items():
        print(f"  {label.capitalize():<10} {count:>3} ({count/len(df)*100:.0f}%)")
    print("\nBackends used: " + ", ".join(f"{k}={v}" for k, v in df["backend"].value_counts().items()))
    print("\nSentiment by topic (avg score):")
    for topic, score in df.groupby("topic")["score"].mean().sort_values().items():
        bar = "#" * int(abs(score) * 20)
        sign = "-" if score < 0 else "+"
        print(f"  {topic:<12} {sign}{abs(score):.3f}  {bar}")
    print(f"\nOutput saved to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
