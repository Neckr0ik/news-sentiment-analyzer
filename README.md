# News Sentiment Analyzer

> **Resumen:** Herramienta en Python que lee feeds RSS en inglés, español y portugués (incluye medios mexicanos) y estima el tono de cada titular por fuente y por tema. El inglés se califica con VADER; el español y el portugués usan la API de Claude si hay `ANTHROPIC_API_KEY`, o un léxico curado sin conexión. Las cifras describen el tono de los titulares, no la opinión del público.

*Part of Giovanni Oliveira's Hybrid Intelligence portfolio — see https://github.com/Neckr0ik*

**Business problem:** a newsroom that takes feeds from several outlets wants to see the emotional tone of incoming news by source and by topic, so it can spot story patterns and how each outlet frames them. In Mexico and Latin America most of those feeds are in Spanish or Portuguese, which English-only tools such as VADER cannot read.

This tool fetches RSS/Atom feeds, scores each headline's tone with a **backend that fits its language**, classifies it by topic, and writes charts, a CSV and a markdown report.

> Inspired by editorial analytics work at a Mexican broadcast network, where a Reuters partnership fed three daily news programs that needed editorial signals from international wire content.

---

## How sentiment is scored

| Language | Default backend | Notes |
|---|---|---|
| English (`en`) | **VADER** (`vaderSentiment`) | Offline, rule-based, tuned for short English text. |
| Spanish (`es`), Portuguese (`pt`) | **Claude API**, *if* `ANTHROPIC_API_KEY` is set | Multilingual; headlines are sent in batches of 20 and come back as structured JSON scores in [-1, 1]. The default model id is set in `src/sentiment.py` (`DEFAULT_CLAUDE_MODEL`); override it with `NEWS_SENTIMENT_CLAUDE_MODEL`. |
| Spanish, Portuguese (no key) | **Offline lexicon** (`src/lexicon.py`) | About 140–160 curated news words per language, with accent folding, negation (`no`, `sin`, `não`, `nem`…), intensifiers (`muy`, `muito`…) and a few multi-word entries (`cae el desempleo`). It uses VADER's normalisation. |

If a Claude call fails (network, rate limit, refusal, malformed output), that batch is scored with the offline backend and the run continues. The `backend` column in `results.csv` records which backend scored each row.

Labels: `positive` if score ≥ 0.05, `negative` if ≤ -0.05, otherwise `neutral` (the VADER convention, used for every backend).

Topics (`Politics`, `Economy`, `Technology`, `Conflict`, `Climate`, `Health`, `Business`, else `General`) come from per-language keyword lists with word-boundary matching.

---

## Feeds

Feeds are configured in [`config/feeds.json`](config/feeds.json). Each entry has a name, URL, language and region. The defaults are:

- **English:** BBC World, BBC Technology, The Guardian World
- **Spanish (Mexico / LATAM):** El Universal, El Financiero, La Jornada, Expansión, BBC Mundo
- **Portuguese (Brazil):** Folha de S.Paulo, g1, BBC News Brasil

Publishers change feed URLs from time to time. A feed that fails is skipped with a warning.

---

## Install

```bash
git clone https://github.com/Neckr0ik/news-sentiment-analyzer.git
cd news-sentiment-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+. The `anthropic` package is installed but only used when `ANTHROPIC_API_KEY` is set.

## Usage

```bash
python main.py                          # all feeds (en, es, pt) -> output/
python main.py --languages es           # Spanish-language feeds only
python main.py --demo --no-charts       # offline: bundled sample headlines, no network
python main.py --input my_articles.json # score your own list of {"title", "language", ...}
export ANTHROPIC_API_KEY=...            # optional: Claude scores es/pt
python main.py --backend claude         # force Claude for every language
python main.py --backend lexicon        # force the offline backends
```

| Output | Content |
|---|---|
| `output/results.csv` | One row per article: source, language, title, topic, sentiment, score, backend, link |
| `output/report.md` | Totals, a language × backend table, and the 5 most negative and 5 most positive headlines |
| `01_sentiment_by_source.png` | Positive/neutral/negative counts per source |
| `02_sentiment_by_topic.png` | Average score per topic |
| `03_score_distribution.png` | Score distribution per topic |
| `04_source_topic_heatmap.png` | Source × topic average score (blank = no articles) |

`data/sample_headlines.json` holds 15 made-up example headlines (5 per language) for the offline demo. They are not real news.

---

## Tests

```bash
pytest -q
```

The tests run fully offline. Feed fetching uses an injected fake HTTP getter. The Claude backend is tested with a fake client and also through the real `anthropic` SDK pointed at a local stub server, so no API key or internet access is needed. CI runs the tests on Python 3.11 and 3.12 (`.github/workflows/tests.yml`).

## Project structure

```
news-sentiment-analyzer/
├── main.py                 # CLI: fetch/load -> score -> CSV, charts, report
├── config/feeds.json       # feed list (en / es / pt)
├── data/sample_headlines.json
├── src/
│   ├── feeds.py            # RSS 2.0 + Atom parsing, HTML cleanup, config loading
│   ├── sentiment.py        # backends (VADER, lexicon, Claude), router, topic classifier
│   ├── lexicon.py          # Spanish/Portuguese lexicons with negation handling
│   └── reporter.py         # charts + markdown report
└── tests/
```

---

## Limitations

- **Tone of headlines, not public opinion.** Scores estimate the wording of headlines and summaries. They say nothing about how audiences feel and are not a survey or audience measurement.
- **The scores do not mean the same thing across backends.** VADER, the lexicon and Claude are different instruments. Compare sources within one backend, and check the `backend` column before you compare languages.
- **The offline lexicon is small and naive.** It misses sarcasm, most compositional meaning ("la inflación baja" is handled; many similar phrases are not) and domain shifts. It has not been validated against a labelled Spanish or Portuguese corpus.
- **VADER was built for English social media.** News headlines are often neutral or mixed, and VADER can misread them.
- **Keyword topics are coarse.** Many articles fall into `General`.
- **LLM scores are not deterministic** and cost money per call. Headlines are sent to the Anthropic API only when you set a key.

## License

MIT © 2026 Giovanni Oliveira
