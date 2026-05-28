# News Sentiment Analyzer

**Business problem:** A newsroom managing feeds from multiple international wire services needs to understand the emotional tone of incoming news — by source, by topic, and over time — to make better editorial decisions and identify emerging story patterns.

This tool fetches live RSS feeds from major news outlets, scores each article's sentiment using VADER NLP, classifies it by topic, and produces a full visual report.

> Inspired by editorial analytics work at UNIFE México, where a Reuters partnership fed three daily news programs requiring real-time editorial signal processing across international wire content.

---

## What it does

| Output | Insight |
|---|---|
| `01_sentiment_by_source.png` | Positive/neutral/negative breakdown per news source |
| `02_sentiment_by_topic.png` | Which topics (Economy, Politics, Conflict…) skew most negative or positive |
| `03_score_distribution.png` | Sentiment score distribution curves by topic |
| `04_source_topic_heatmap.png` | Source × topic sentiment matrix — spot editorial bias at a glance |
| `output/report.md` | Top 5 most positive and most negative headlines with scores |
| `output/results.csv` | Full scored dataset for downstream analysis |

---

## Quick start

```bash
git clone https://github.com/Neckr0ik/news-sentiment-analyzer.git
cd news-sentiment-analyzer
pip install -r requirements.txt
python main.py
```

Fetches live data — requires internet connection. Runtime: ~15 seconds.

---

## Project structure

```
news-sentiment-analyzer/
├── main.py              # Entry point — fetch, analyze, report
├── src/
│   ├── feeds.py         # RSS feed fetcher (Reuters, BBC, AP)
│   ├── sentiment.py     # VADER scoring + keyword-based topic classifier
│   └── reporter.py      # Chart generation + markdown report writer
├── output/              # Generated charts and report (git-ignored)
└── requirements.txt
```

---

## Extending with an LLM

The `sentiment.py` module is designed to accept an optional Claude API call for deeper analysis (narrative framing, bias detection, editorial risk flags). To enable:

```python
import anthropic
client = anthropic.Anthropic()
# Pass article text to claude-haiku-4-5 for sub-second per-article analysis
```

---

## Skills demonstrated

`Python` · `NLP` · `VADER Sentiment` · `RSS/XML Parsing` · `Pandas` · `Matplotlib` · `Seaborn` · `News Analytics` · `Topic Classification`

---

## License

MIT © 2026 Giovanni Oliveira
