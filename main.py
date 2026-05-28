"""
News Sentiment Analyzer
Run:  python main.py
Fetches live RSS feeds, scores sentiment with VADER, outputs charts + markdown report.
"""
import os
import pandas as pd
from src.feeds import fetch_all
from src.sentiment import analyze_batch
from src.reporter import (plot_sentiment_by_source, plot_sentiment_by_topic,
                           plot_score_distribution, plot_heatmap, write_markdown_report)

def main():
    print("News Sentiment Analyzer")
    print("=" * 50)

    print("\n[1/3] Fetching news feeds...")
    articles = fetch_all(max_per_feed=15)
    print(f"  Total articles fetched: {len(articles)}")

    if not articles:
        print("  No articles fetched. Check your internet connection.")
        return

    print("\n[2/3] Running sentiment analysis...")
    results = analyze_batch(articles)
    df = pd.DataFrame(results)

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/results.csv", index=False, encoding="utf-8")
    print(f"  Analyzed {len(df)} articles across {df['source'].nunique()} sources")

    print("\n[3/3] Generating charts and report...")
    plot_sentiment_by_source(df)
    plot_sentiment_by_topic(df)
    plot_score_distribution(df)
    plot_heatmap(df)
    write_markdown_report(df, "output/report.md")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Articles:  {len(df)}")
    for label, count in df["sentiment"].value_counts().items():
        print(f"  {label.capitalize():<10} {count:>3} ({count/len(df)*100:.0f}%)")
    print(f"\nSentiment by topic (avg score):")
    for topic, score in df.groupby("topic")["score"].mean().sort_values().items():
        bar = "#" * int(abs(score) * 20)
        sign = "-" if score < 0 else "+"
        print(f"  {topic:<12} {sign}{abs(score):.3f}  {bar}")
    print("\nOutput saved to output/")

if __name__ == "__main__":
    main()
