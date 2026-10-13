"""Generates charts and a markdown summary report from sentiment results."""
from __future__ import annotations

import os
import re
from urllib.parse import quote, urlsplit

import matplotlib
matplotlib.use("Agg")  # headless: works in CI and over SSH
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
DEFAULT_OUT = "output"

_MD_SPECIAL = re.compile(r"([\\`*_\[\]()!|#])")
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def md_text(value) -> str:
    """Escape markdown/HTML syntax so feed text can't add links, images or tags to the report."""
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = _MD_SPECIAL.sub(r"\\\1", text)
    return text.replace("<", "&lt;").replace(">", "&gt;")


def safe_link(url) -> str:
    """Return the URL ready for a markdown link, or "" unless it is an absolute http(s) URL."""
    url = str(url or "").strip()
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return ""
    # percent-encode anything that could end the link target early or open markup
    return quote(url, safe=":/?#@!$&'*+,;=%~[]-._")


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df where text cells starting with = + - @ (or tab/CR) get a leading quote,
    so spreadsheet apps show them as text instead of running them as formulas."""
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "'" + v if isinstance(v, str) and v.startswith(_CSV_FORMULA_PREFIXES) else v)
    return out


def _save(out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_sentiment_by_source(df: pd.DataFrame, out_dir: str = DEFAULT_OUT) -> str:
    counts = df.groupby(["source", "sentiment"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    counts.plot(kind="barh", stacked=True, figsize=(10, max(4, 0.45 * len(counts) + 1)),
                color=["#2E8B57", "#888888", "#C0392B"])
    plt.xlabel("Article Count")
    plt.title("Sentiment Distribution by News Source", fontweight="bold")
    return _save(out_dir, "01_sentiment_by_source.png")


def plot_sentiment_by_topic(df: pd.DataFrame, out_dir: str = DEFAULT_OUT) -> str:
    avg = df.groupby("topic")["score"].mean().sort_values()
    colors = ["#C0392B" if v < 0 else "#2E8B57" for v in avg]
    avg.plot(kind="barh", color=colors, figsize=(9, 5))
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Average Sentiment Score (compound, -1 to 1)")
    plt.title("Average Sentiment Score by Topic", fontweight="bold")
    return _save(out_dir, "02_sentiment_by_topic.png")


def plot_score_distribution(df: pd.DataFrame, out_dir: str = DEFAULT_OUT) -> str:
    plt.figure(figsize=(9, 4))
    for topic, grp in df.groupby("topic"):
        if grp["score"].nunique() > 1:  # KDE is undefined for a single distinct value
            sns.kdeplot(grp["score"], label=topic, fill=False, warn_singular=False)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Sentiment Score")
    plt.title("Sentiment Score Distribution by Topic", fontweight="bold")
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(fontsize=8)
    return _save(out_dir, "03_score_distribution.png")


def plot_heatmap(df: pd.DataFrame, out_dir: str = DEFAULT_OUT) -> str:
    # Missing source x topic cells stay blank (NaN) instead of being shown as a neutral 0.
    pivot = df.groupby(["source", "topic"])["score"].mean().unstack()
    plt.figure(figsize=(12, max(4, 0.45 * len(pivot) + 1)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Avg Sentiment Score: Source x Topic", fontweight="bold")
    return _save(out_dir, "04_source_topic_heatmap.png")


def plot_all(df: pd.DataFrame, out_dir: str = DEFAULT_OUT) -> list[str]:
    return [plot_sentiment_by_source(df, out_dir), plot_sentiment_by_topic(df, out_dir),
            plot_score_distribution(df, out_dir), plot_heatmap(df, out_dir)]


def _headline_line(row) -> str:
    title = md_text(str(row["title"])[:80])
    link = safe_link(row.get("link", ""))
    head = f"[{title}]({link})" if link else title
    return f"- {head} `{row['score']:+.3f}` ({md_text(row['source'])})\n"


def write_markdown_report(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    total = len(df)
    pos = int((df["sentiment"] == "positive").sum())
    neg = int((df["sentiment"] == "negative").sum())
    neu = int((df["sentiment"] == "neutral").sum())
    by_topic = df.groupby("topic")["score"].mean()

    lines = [
        "# News Sentiment Analysis Report\n",
        f"**Articles analyzed:** {total}  |  "
        f"**Positive:** {pos} ({pos/total*100:.0f}%)  |  "
        f"**Neutral:** {neu} ({neu/total*100:.0f}%)  |  "
        f"**Negative:** {neg} ({neg/total*100:.0f}%)\n",
        f"\n**Most negative topic:** {by_topic.idxmin()}  |  "
        f"**Most positive topic:** {by_topic.idxmax()}\n",
        "\n## By language and backend\n",
        "\n| Language | Backend | Articles | Avg score |\n|:---|:---|---:|---:|\n",
    ]
    for (lang, backend), grp in df.groupby(["language", "backend"]):
        lines.append(f"| {md_text(lang)} | {md_text(backend)} | {len(grp)} | {grp['score'].mean():+.3f} |\n")

    lines.append("\n## Top 5 Most Negative Headlines\n")
    lines.extend(_headline_line(row) for _, row in df.nsmallest(5, "score").iterrows())
    lines.append("\n## Top 5 Most Positive Headlines\n")
    lines.extend(_headline_line(row) for _, row in df.nlargest(5, "score").iterrows())
    lines.append("\n---\n*Scores describe headline tone as estimated by the listed backend "
                 "(VADER, an offline lexicon, or an LLM). They are not a measure of audience opinion.*\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return path
