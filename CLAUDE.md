# CLAUDE.md — news-sentiment-analyzer

Portfolio project (Neckr0ik / Giovanni Oliveira, "Hybrid Intelligence" portfolio). Fetches RSS/Atom feeds in English, Spanish and Portuguese (Mexican/LATAM and Brazilian outlets included), scores each headline's tone with a language-appropriate backend, classifies topics, and writes `output/results.csv`, `output/report.md` and 4 charts. Measures headline tone, not public opinion.

## Stack

Python 3.11+, vaderSentiment, pandas, matplotlib, seaborn, defusedxml, anthropic (optional backend), pytest (`requirements.txt`).

Backends (`src/sentiment.py`):
- `en` → VADER (offline).
- `es`/`pt` → Claude API **only if `ANTHROPIC_API_KEY` is set** (batches of 20, structured JSON scores in [-1, 1]); model id from `DEFAULT_CLAUDE_MODEL`, overridable with env `NEWS_SENTIMENT_CLAUDE_MODEL`.
- `es`/`pt` without key → offline lexicon (`src/lexicon.py`) with accent folding, intensifiers and negation.

## Structure

```
main.py                     # CLI: fetch/load -> score -> CSV, charts, report
config/feeds.json           # feed list (name, url, language, region)
data/sample_headlines.json  # 15 made-up headlines for --demo
src/feeds.py                # RSS 2.0 + Atom parsing (defusedxml), HTML cleanup, config loading
src/sentiment.py            # VADER / lexicon / Claude backends, router, topic classifier
src/lexicon.py              # es/pt lexicons, negation rules
src/reporter.py             # charts + markdown report
tests/                      # conftest.py (fake Claude client), feeds, lexicon, sentiment, reporter, cli, SDK stub server
```

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                          # all feeds (network) -> output/
python main.py --languages es           # subset of en,es,pt
python main.py --demo --no-charts       # offline, bundled sample headlines
python main.py --input my_articles.json # score a local JSON list of {"title", "language", ...}
python main.py --backend claude         # auto | vader | lexicon | claude (claude needs ANTHROPIC_API_KEY)
python main.py --backend lexicon        # force offline backends
python main.py --help                   # also: --feeds PATH, --max-per-feed N (15), --out DIR
pytest -q                               # 61 tests, fully offline
```

## Gotchas

- Claude failure handling: transient errors (network, rate limit, 5xx, refusal, malformed output) fall back to the offline backend **for that batch only**. Request errors (HTTP 400/401/403/404) make `--backend claude` exit with code 3; in `auto` Claude is disabled for the rest of the run. The `backend` column in results.csv records who scored each row.
- Labels: positive ≥ 0.05, negative ≤ −0.05, else neutral (all backends).
- Topic = most keyword hits, ties go to the first topic in order (Politics, Economy, Technology, Conflict, Climate, Health, Business, else General). Bare Spanish "partido" is intentionally not Politics.
- Security hardening to preserve: defusedxml (no entities/external entities), responses > 5 MB skipped, CSV formula-injection guard (leading `'` for cells starting with `= + - @`), Markdown/HTML escaping and http/https-only links in report.md.
- Tests: feed fetching uses an injected fake HTTP getter; the Claude backend is tested with a fake client and with the real `anthropic` SDK against a local stub server. Never make tests hit the network or need a real key.
- Feed URLs change; failing feeds are skipped with a warning.
- Never commit an API key; `.env` is gitignored.

## Conventions (shared across the owner's portfolio repos)

- Plain Python 3.11+ CLI: `main.py` (argparse) at the root, logic in `src/`, tests in `tests/`. No package build, no lint/format config.
- `pytest.ini` sets `testpaths = tests` and `pythonpath = .`, so run `pytest` from the repo root.
- CI (`.github/workflows/tests.yml`): `pip install -r requirements.txt` + `pytest -q` on Python 3.11 and 3.12, on every push and PR. Tests must stay offline (no network, no API keys).
- Generated files go to `output/` (gitignored). Never commit them.
- README is English with a Spanish `> **Resumen:**` blurb at the top; it documents formulas and limitations precisely. When you change behaviour, update the README in the same change (tests often pin the documented formulas).
- Be honest about scope: the README's "Limitations" section is part of the product. Don't add claims the code doesn't support.

## Memoria persistente (Graphiti MCP)
Este proyecto usa el connector **Graphiti** (memoria en grafo compartida entre sesiones).
- `group_id` de este repo: `news-sentiment-analyzer`. Datos del usuario y transversales: `neckr0ik-global`.
- Al empezar una tarea: busca contexto con `search_memory_facts` y `search_nodes` usando `group_ids: ["news-sentiment-analyzer", "neckr0ik-global"]`.
- Al terminar algo relevante (decisión de arquitectura, convención nueva, bug resuelto y su causa, preferencia del usuario, estado de un despliegue): guárdalo con `add_memory` (`group_id: "news-sentiment-analyzer"`, `source: "text"`) con un nombre descriptivo.
- Nunca guardes secretos, API keys, tokens, contraseñas ni datos personales sensibles.
- Si Graphiti no está disponible en la sesión, continúa normalmente.
