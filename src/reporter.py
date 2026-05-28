"""Generates charts and a markdown summary report from sentiment results."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
OUT = os.path.join(os.path.dirname(__file__), "..", "output")

def _save(name: str):
    os.makedirs(OUT, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close()

def plot_sentiment_by_source(df: pd.DataFrame):
    counts = df.groupby(["source", "sentiment"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    counts.plot(kind="barh", stacked=True, figsize=(10, 5),
                color=["#2E8B57", "#888888", "#C0392B"])
    plt.xlabel("Article Count")
    plt.title("Sentiment Distribution by News Source", fontweight="bold")
    _save("01_sentiment_by_source.png")

def plot_sentiment_by_topic(df: pd.DataFrame):
    avg = df.groupby("topic")["score"].mean().sort_values()
    colors = ["#C0392B" if v < 0 else "#2E8B57" for v in avg]
    avg.plot(kind="barh", color=colors, figsize=(9, 5))
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Average Sentiment Score (VADER compound)")
    plt.title("Average Sentiment Score by Topic", fontweight="bold")
    _save("02_sentiment_by_topic.png")

def plot_score_distribution(df: pd.DataFrame):
    plt.figure(figsize=(9, 4))
    for topic, grp in df.groupby("topic"):
        sns.kdeplot(grp["score"], label=topic, fill=False)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Sentiment Score")
    plt.title("Sentiment Score Distribution by Topic", fontweight="bold")
    plt.legend(fontsize=8)
    _save("03_score_distribution.png")

def plot_heatmap(df: pd.DataFrame):
    pivot = df.groupby(["source", "topic"])["score"].mean().unstack(fill_value=0)
    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5)
    plt.title("Avg Sentiment Score: Source x Topic", fontweight="bold")
    _save("04_source_topic_heatmap.png")

def write_markdown_report(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total = len(df)
    pos = (df["sentiment"] == "positive").sum()
    neg = (df["sentiment"] == "negative").sum()
    neu = (df["sentiment"] == "neutral").sum()
    most_neg_topic = df.groupby("topic")["score"].mean().idxmin()
    most_pos_topic = df.groupby("topic")["score"].mean().idxmax()

    lines = [
        "# News Sentiment Analysis Report\n",
        f"**Articles analyzed:** {total}  |  "
        f"**Positive:** {pos} ({pos/total*100:.0f}%)  |  "
        f"**Neutral:** {neu} ({neu/total*100:.0f}%)  |  "
        f"**Negative:** {neg} ({neg/total*100:.0f}%)\n",
        f"\n**Most negative topic:** {most_neg_topic}  |  "
        f"**Most positive topic:** {most_pos_topic}\n",
        "\n## Top 5 Most Negative Headlines\n",
    ]
    for _, row in df.nsmallest(5, "score").iterrows():
        lines.append(f"- [{row['title'][:80]}]({row['link']}) `{row['score']:+.3f}`\n")
    lines.append("\n## Top 5 Most Positive Headlines\n")
    for _, row in df.nlargest(5, "score").iterrows():
        lines.append(f"- [{row['title'][:80]}]({row['link']}) `{row['score']:+.3f}`\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  report saved -> {path}")
