"""
Pluggable sentiment backends + keyword topic classifier.

Backends (all return a compound score in [-1, 1]):
  - "vader"   : VADER, English only, offline.
  - "lexicon" : small curated Spanish/Portuguese lexicon with negation, offline.
  - "claude"  : optional, multilingual, via the Anthropic API. Used only when
                ANTHROPIC_API_KEY is set (or a client is injected, e.g. in tests).

Backend selection with backend="auto" (the default):
  English -> VADER; Spanish/Portuguese -> Claude if ANTHROPIC_API_KEY is set,
  otherwise the offline lexicon. backend="claude" forces Claude for every language.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterable, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import lexicon

POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05
SUPPORTED_LANGUAGES = ("en", "es", "pt")
DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"
CLAUDE_BATCH_SIZE = 20


def label_for(score: float) -> str:
    if score >= POS_THRESHOLD:
        return "positive"
    if score <= NEG_THRESHOLD:
        return "negative"
    return "neutral"


# ── Language detection (tiny stopword heuristic; feeds also declare a language) ──

_STOPWORDS = {
    "en": {"the", "and", "of", "to", "in", "is", "for", "on", "with", "as", "at", "by",
           "from", "after", "over", "says", "will", "has", "was", "are"},
    "es": {"el", "la", "los", "las", "de", "del", "y", "en", "que", "por", "para", "con",
           "una", "un", "se", "su", "al", "es", "tras", "sobre", "como", "mas"},
    "pt": {"o", "a", "os", "as", "de", "do", "da", "dos", "das", "e", "em", "que", "para",
           "com", "uma", "um", "no", "na", "nos", "nas", "ao", "pelo", "pela", "apos", "nao"},
}


def detect_language(text: str, default: str = "en") -> str:
    tokens = lexicon.tokenize(text)
    if not tokens:
        return default
    counts = {lang: sum(t in words for t in tokens) for lang, words in _STOPWORDS.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else default


# ── Topic classifier (word-boundary keyword match, per language) ──────────────

TOPIC_KEYWORDS = {
    "en": {
        "Politics":   ["election", "president", "government", "minister", "congress", "senate", "vote", "policy"],
        "Economy":    ["inflation", "gdp", "market", "markets", "stock", "trade", "tariff", "tariffs", "recession", "fed", "bank"],
        "Technology": ["ai", "artificial intelligence", "tech", "software", "cyber", "digital", "startup"],
        "Conflict":   ["war", "military", "attack", "conflict", "troops", "ceasefire", "missile", "bomb"],
        "Climate":    ["climate", "carbon", "emission", "emissions", "flood", "floods", "wildfire", "wildfires",
                       "renewable", "hurricane"],
        "Health":     ["health", "hospital", "vaccine", "disease", "pandemic", "drug", "cancer", "medicine"],
        "Business":   ["merger", "acquisition", "revenue", "profit", "ceo", "company", "earnings", "deal"],
    },
    "es": {
        "Politics":   ["elección", "elecciones", "presidente", "presidenta", "gobierno", "senado", "diputados",
                       "congreso", "reforma", "votación", "partido político", "partidos políticos"],
        "Economy":    ["inflación", "pib", "mercado", "mercados", "bolsa", "aranceles", "banxico", "economía",
                       "recesión", "dólar", "tasas"],
        "Technology": ["ia", "inteligencia artificial", "tecnología", "ciberseguridad", "digital", "software"],
        "Conflict":   ["guerra", "militar", "ataque", "conflicto", "tropas", "misil", "cártel", "narco"],
        "Climate":    ["clima", "huracán", "inundación", "inundaciones", "incendio", "sequía", "lluvias", "sismo"],
        "Health":     ["salud", "hospital", "vacuna", "enfermedad", "imss", "medicamentos", "cáncer"],
        "Business":   ["empresa", "empresas", "ventas", "ganancias", "inversión", "fusión", "utilidades"],
    },
    "pt": {
        "Politics":   ["eleição", "eleições", "presidente", "governo", "ministro", "senado", "câmara", "votação"],
        "Economy":    ["inflação", "pib", "mercado", "bolsa", "juros", "economia", "dólar", "recessão"],
        "Technology": ["ia", "inteligência artificial", "tecnologia", "aplicativo", "digital", "software"],
        "Conflict":   ["guerra", "militar", "ataque", "conflito", "tropas", "míssil", "tiroteio"],
        "Climate":    ["clima", "furacão", "enchente", "enchentes", "queimadas", "seca", "chuvas"],
        "Health":     ["saúde", "hospital", "vacina", "doença", "medicamentos", "câncer"],
        "Business":   ["empresa", "empresas", "vendas", "lucro", "investimento", "fusão"],
    },
}

_TOPIC_PATTERNS = {
    lang: {topic: [re.compile(r"\b" + re.escape(lexicon.fold(kw)) + r"\b") for kw in kws]
           for topic, kws in topics.items()}
    for lang, topics in TOPIC_KEYWORDS.items()
}


def classify_topic(text: str, language: str = "en") -> str:
    """Keyword topic for a headline; returns 'General' when nothing matches.

    The topic with the most matching keywords wins. Ties go to the topic listed first in
    TOPIC_KEYWORDS (Politics, Economy, Technology, Conflict, Climate, Health, Business),
    so the result is deterministic. A bare "partido" is not a Politics keyword because in
    Spanish news it usually means a football match; only "partido político" counts.
    """
    patterns = _TOPIC_PATTERNS.get(language, _TOPIC_PATTERNS["en"])
    folded = lexicon.fold(text)
    scores = {topic: sum(bool(p.search(folded)) for p in pats) for topic, pats in patterns.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


# ── Backends ──────────────────────────────────────────────────────────────────

class VaderBackend:
    name = "vader"
    languages = {"en"}

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    def score_batch(self, texts: list[str], language: str = "en") -> list[float]:
        return [self._vader.polarity_scores(t)["compound"] for t in texts]


class LexiconBackend:
    name = "lexicon"
    languages = {"es", "pt"}

    def score_batch(self, texts: list[str], language: str = "es") -> list[float]:
        return [lexicon.score_text(t, language)["compound"] for t in texts]


class ClaudeBackendError(RuntimeError):
    """Claude answered, but the answer could not be used (refusal, truncation, bad JSON)."""


class ClaudeConfigError(RuntimeError):
    """The request itself is rejected (bad key, no permission, unknown model, bad request).

    Retrying or falling back per chunk would not help, so a forced ``--backend claude``
    run stops with this error instead of silently scoring everything offline.
    """


FATAL_STATUS_CODES = {400, 401, 403, 404}


def _recoverable_errors() -> tuple:
    """Errors after which one chunk is rescored offline and the run continues."""
    errors: tuple = (ClaudeBackendError, json.JSONDecodeError, KeyError)
    try:
        import anthropic
    except ImportError:
        return errors
    return errors + (anthropic.APIStatusError, anthropic.APIConnectionError)


_CLAUDE_SYSTEM = (
    "You rate the tone of news headlines for a media-analytics tool. For each item, "
    "return a score from -1.0 (clearly negative events or framing) to 1.0 (clearly positive), "
    "with values near 0 for neutral or purely informational items. Judge the headline as written, "
    "in its own language (English, Spanish or Portuguese); do not add outside facts. "
    "Return exactly one result per input id."
)

_CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["id", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


class ClaudeBackend:
    """Multilingual scoring through the Anthropic Messages API with structured JSON output."""
    name = "claude"
    languages = set(SUPPORTED_LANGUAGES)

    def __init__(self, client=None, model: Optional[str] = None, batch_size: int = CLAUDE_BATCH_SIZE):
        if client is None:
            import anthropic  # imported lazily so offline use never needs the package configured
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("NEWS_SENTIMENT_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
        self.batch_size = batch_size

    def _score_chunk(self, texts: list[str], language: str) -> list[float]:
        payload = json.dumps([{"id": i, "text": t} for i, t in enumerate(texts)], ensure_ascii=False)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=_CLAUDE_SYSTEM,
            messages=[{"role": "user", "content": f"Language hint: {language}\nItems:\n{payload}"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _CLAUDE_SCHEMA}},
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ClaudeBackendError(f"response stopped with stop_reason={response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise ClaudeBackendError("no text block in response")
        try:
            by_id = {int(r["id"]): float(r["score"]) for r in json.loads(text)["results"]}
        except (KeyError, TypeError, ValueError) as e:  # JSONDecodeError is a ValueError
            raise ClaudeBackendError(f"malformed JSON output ({type(e).__name__}: {e})") from e
        missing = [i for i in range(len(texts)) if i not in by_id]
        if missing:
            raise ClaudeBackendError(f"missing scores for ids {missing}")
        return [max(-1.0, min(1.0, by_id[i])) for i in range(len(texts))]

    def score_batch(self, texts: list[str], language: str = "en") -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            scores.extend(self._score_chunk(texts[start:start + self.batch_size], language))
        return scores


def claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class BackendRouter:
    """Chooses a backend per language and falls back offline if Claude fails.

    Fallback is per chunk: chunks Claude scored are kept, and only a failed chunk is
    rescored offline. Auth/request errors (HTTP 400/401/403/404) stop a forced
    ``mode="claude"`` run with ClaudeConfigError; in ``auto`` mode they switch Claude off
    for the rest of the run and the offline backends take over.
    """

    def __init__(self, mode: str = "auto", claude_client=None, claude_model: Optional[str] = None):
        if mode not in ("auto", "vader", "lexicon", "claude"):
            raise ValueError(f"unknown backend mode: {mode}")
        self.mode = mode
        self.vader = VaderBackend()
        self.lexicon = LexiconBackend()
        self.claude = None
        if mode in ("auto", "claude") and (claude_client is not None or claude_available()):
            self.claude = ClaudeBackend(client=claude_client, model=claude_model)
        elif mode == "claude":
            raise RuntimeError("backend 'claude' requires ANTHROPIC_API_KEY to be set")

    def offline_for(self, language: str):
        return self.vader if language == "en" else self.lexicon

    def backend_for(self, language: str):
        if self.mode == "vader":
            return self.vader
        if self.mode == "lexicon":
            return self.lexicon if language in self.lexicon.languages else self.vader
        if self.mode == "claude":
            return self.claude
        # auto
        if language == "en":
            return self.vader
        return self.claude or self.lexicon

    def score(self, texts: list[str], language: str) -> tuple[list[float], list[str]]:
        """Return (scores, backend name per text)."""
        backend = self.backend_for(language)
        if backend is not self.claude:
            return backend.score_batch(texts, language), [backend.name] * len(texts)

        recoverable = _recoverable_errors()
        offline = self.offline_for(language)
        scores: list[float] = []
        names: list[str] = []
        for start in range(0, len(texts), backend.batch_size):
            chunk = texts[start:start + backend.batch_size]
            if self.claude is not None:
                try:
                    scores.extend(backend.score_batch(chunk, language))
                    names.extend([backend.name] * len(chunk))
                    continue
                except recoverable as e:
                    status = getattr(e, "status_code", None)
                    if status in FATAL_STATUS_CODES:
                        message = (f"Claude API rejected the request (HTTP {status}: {type(e).__name__}). "
                                   "Check ANTHROPIC_API_KEY and NEWS_SENTIMENT_CLAUDE_MODEL.")
                        if self.mode == "claude":
                            raise ClaudeConfigError(message) from e
                        print(f"  [warn] {message} Using the offline backends for the rest of the run.")
                        self.claude = None
                    else:
                        print(f"  [warn] Claude backend failed on {len(chunk)} items "
                              f"({type(e).__name__}: {e}); scoring them offline")
            scores.extend(offline.score_batch(chunk, language))
            names.extend([offline.name] * len(chunk))
        return scores, names


def analyze_batch(articles: Iterable, backend: str = "auto", router: Optional[BackendRouter] = None) -> list[dict]:
    articles = list(articles)
    router = router or BackendRouter(mode=backend)

    langs = []
    for art in articles:
        lang = (getattr(art, "language", "") or "").lower()[:2]
        langs.append(lang if lang in SUPPORTED_LANGUAGES else detect_language(art.full_text))

    scores: dict[int, tuple[float, str]] = {}
    for lang in sorted(set(langs)):
        idx = [i for i, l in enumerate(langs) if l == lang]
        values, used = router.score([articles[i].full_text for i in idx], lang)
        for i, v, name in zip(idx, values, used):
            scores[i] = (v, name)

    results = []
    for i, art in enumerate(articles):
        score, used = scores[i]
        results.append({
            "source":    art.source,
            "language":  langs[i],
            "title":     art.title,
            "topic":     classify_topic(art.full_text, langs[i]),
            "sentiment": label_for(score),
            "score":     round(score, 4),
            "backend":   used,
            "link":      art.link,
        })
    return results
