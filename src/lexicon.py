"""
Small, curated news-tone lexicons for Spanish and Portuguese (offline fallback).

This is deliberately simple and transparent: a few hundred hand-picked words
with valences in [-3, 3], plus VADER-style negation and intensifier handling,
normalised to a compound score in [-1, 1]. It is good enough to rank headlines
by tone and to run fully offline; it is NOT a validated sentiment model.
Accents are folded, so "crisis"/"crísis" and "não"/"nao" match the same entry.
"""
from __future__ import annotations

import math
import re
import unicodedata

# ── Lexicons (accent-folded, lowercase). Valence: -3 very negative … +3 very positive.
ES_LEXICON: dict[str, float] = {
    # violence / disaster / death
    "muere": -2.5, "mueren": -2.5, "fallece": -2.5, "fallecen": -2.5, "quito la vida": -3,
    "muerto": -3, "muertos": -3, "muerta": -3, "muertas": -3, "muerte": -3, "muertes": -3,
    "asesinato": -3, "asesinatos": -3, "asesinado": -3, "asesinada": -3, "homicidio": -3,
    "homicidios": -3, "masacre": -3, "feminicidio": -3, "secuestro": -3, "secuestrado": -3,
    "violencia": -2.5, "violento": -2.5, "ataque": -2.5, "ataques": -2.5, "atentado": -3,
    "balacera": -3, "enfrentamiento": -2, "guerra": -3, "bombardeo": -3, "victimas": -2.5,
    "victima": -2.5, "herido": -2, "heridos": -2, "desaparecido": -2.5, "desaparecidos": -2.5,
    "tragedia": -3, "desastre": -2.5, "catastrofe": -3, "huracan": -1.5, "sismo": -1.5,
    "terremoto": -2, "inundacion": -2, "inundaciones": -2, "incendio": -2, "danos": -1.5,
    "danadas": -1.5, "danados": -1.5, "destruccion": -2.5, "emergencia": -1.5,
    # crime / politics / institutions
    "corrupcion": -2.5, "fraude": -2.5, "delito": -2, "crimen": -2.5, "detenido": -1,
    "detenidos": -1, "denuncia": -1, "acusa": -1.5, "acusado": -1.5, "escandalo": -2,
    "protesta": -1, "protestas": -1, "conflicto": -2, "amenaza": -2, "amenazas": -2,
    "crisis": -2, "polemica": -1, "condena": -1.5, "rechaza": -1, "rechazo": -1,
    # economy (negative)
    "inflacion": -1, "recesion": -2.5, "desempleo": -2, "deuda": -1, "quiebra": -2.5,
    "caida": -1.5, "cae": -1.5, "caen": -1.5, "desplome": -2.5, "se desploma": -2.5,
    "perdida": -1.5, "perdidas": -1.5, "pierde": -1.5, "pierden": -1.5, "recorte": -1.5,
    "recortes": -1.5, "despidos": -2, "pobreza": -2, "escasez": -2, "riesgo": -1,
    "preocupacion": -1.5, "temor": -1.5, "miedo": -2, "alerta": -1.5, "falla": -1.5,
    # general negative
    "malo": -2, "mala": -2, "peor": -2.5, "grave": -2, "dificil": -1, "problema": -1.5,
    "problemas": -1.5, "fracaso": -2.5, "fracasa": -2.5, "triste": -2, "dolor": -2,
    "enfermedad": -1.5, "epidemia": -2, "brote": -1.5,
    # positive
    "bueno": 2, "buena": 2, "buenos": 2, "buenas": 2, "mejor": 2, "mejora": 2, "mejoras": 2,
    "excelente": 3, "exito": 2.5, "exitoso": 2.5, "logro": 2.5, "logra": 2, "logran": 2,
    "gana": 2, "ganan": 2, "victoria": 2.5, "triunfo": 2.5, "celebra": 2, "celebran": 2,
    "acuerdo": 1.5, "avance": 1.5, "avances": 1.5, "crecimiento": 1.5, "crece": 1.5,
    "recuperacion": 2, "se recupera": 2, "repunte": 1.5, "record": 1, "inversion": 1,
    "empleo": 1, "empleos": 1, "beneficio": 1.5, "beneficios": 1.5, "apoyo": 1.5,
    "ayuda": 1.5, "rescate": 1.5, "rescatan": 2, "paz": 2.5, "esperanza": 2, "alegria": 3,
    "feliz": 2.5, "innovacion": 1.5, "premio": 2, "reconocimiento": 2, "estabilidad": 1.5,
    "salva": 2, "solucion": 1.5, "oportunidad": 1.5, "fortalece": 1.5,
    "desempleo cae": 1.5, "baja la inflacion": 1.5, "inflacion baja": 1.5, "cae la inflacion": 1.5,
    "cae el desempleo": 1.5,
}

PT_LEXICON: dict[str, float] = {
    "morre": -2.5, "morrem": -2.5, "falece": -2.5, "faleceu": -2.5, "morreu": -2.5,
    "morto": -3, "mortos": -3, "morta": -3, "mortas": -3, "morte": -3, "mortes": -3,
    "assassinato": -3, "assassinado": -3, "homicidio": -3, "chacina": -3, "massacre": -3,
    "sequestro": -3, "violencia": -2.5, "violento": -2.5, "ataque": -2.5, "ataques": -2.5,
    "atentado": -3, "tiroteio": -3, "guerra": -3, "bombardeio": -3, "vitimas": -2.5,
    "vitima": -2.5, "ferido": -2, "feridos": -2, "desaparecido": -2.5, "tragedia": -3,
    "desastre": -2.5, "catastrofe": -3, "furacao": -1.5, "terremoto": -2, "enchente": -2,
    "enchentes": -2, "incendio": -2, "queimadas": -2, "destruicao": -2.5, "emergencia": -1.5,
    "corrupcao": -2.5, "fraude": -2.5, "crime": -2.5, "preso": -1, "presos": -1,
    "denuncia": -1, "acusa": -1.5, "acusado": -1.5, "escandalo": -2, "protesto": -1,
    "protestos": -1, "conflito": -2, "ameaca": -2, "ameacas": -2, "crise": -2,
    "polemica": -1, "condena": -1.5, "rejeita": -1,
    "inflacao": -1, "recessao": -2.5, "desemprego": -2, "divida": -1, "falencia": -2.5,
    "queda": -1.5, "cai": -1.5, "caem": -1.5, "despenca": -2.5, "prejuizo": -2,
    "perda": -1.5, "perdas": -1.5, "perde": -1.5, "corte": -1, "cortes": -1.5,
    "demissoes": -2, "pobreza": -2, "escassez": -2, "risco": -1, "preocupacao": -1.5,
    "medo": -2, "alerta": -1.5, "falha": -1.5,
    "ruim": -2, "pior": -2.5, "grave": -2, "dificil": -1, "problema": -1.5,
    "problemas": -1.5, "fracasso": -2.5, "triste": -2, "dor": -2, "doenca": -1.5,
    "epidemia": -2, "surto": -1.5,
    "bom": 2, "boa": 2, "bons": 2, "boas": 2, "melhor": 2, "melhora": 2, "excelente": 3,
    "sucesso": 2.5, "conquista": 2.5, "conquistas": 2.5, "vence": 2, "vitoria": 2.5,
    "triunfo": 2.5, "comemora": 2, "celebra": 2, "acordo": 1.5, "avanco": 1.5,
    "avancos": 1.5, "crescimento": 1.5, "cresce": 1.5, "recuperacao": 2, "alta": 0.5,
    "recorde": 1, "investimento": 1, "emprego": 1, "empregos": 1, "beneficio": 1.5,
    "apoio": 1.5, "ajuda": 1.5, "resgate": 1.5, "paz": 2.5, "esperanca": 2, "alegria": 3,
    "feliz": 2.5, "inovacao": 1.5, "premio": 2, "estabilidade": 1.5,
    "salva": 2, "solucao": 1.5, "oportunidade": 1.5, "fortalece": 1.5,
    "desemprego cai": 1.5, "inflacao cai": 1.5, "cai a inflacao": 1.5, "cai o desemprego": 1.5,
}

NEGATIONS = {
    "es": {"no", "nunca", "jamas", "sin", "ni", "tampoco", "nadie", "ningun", "ninguna"},
    "pt": {"nao", "nunca", "jamais", "sem", "nem", "ninguem", "nenhum", "nenhuma"},
}
INTENSIFIERS = {
    "es": {"muy": 0.3, "sumamente": 0.4, "extremadamente": 0.4, "tan": 0.2,
           "gran": 0.2, "fuerte": 0.2, "enorme": 0.3},
    "pt": {"muito": 0.3, "extremamente": 0.4, "tao": 0.2, "grande": 0.2,
           "forte": 0.2, "enorme": 0.3},
}
LEXICONS = {"es": ES_LEXICON, "pt": PT_LEXICON}

NEGATION_FACTOR = -0.74   # same dampened flip VADER uses
NEGATION_WINDOW = 3       # tokens after a negator whose valence is flipped
ALPHA = 15                # VADER normalisation constant

_WORD_RE = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    """Lowercase and strip diacritics: 'Recuperación' -> 'recuperacion'."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(fold(text))


def score_text(text: str, language: str) -> dict:
    """Return {'compound': float in [-1, 1], 'matches': [(token, valence), ...]}."""
    lexicon = LEXICONS[language]
    negators = NEGATIONS[language]
    boosters = INTENSIFIERS[language]
    tokens = tokenize(text)

    total = 0.0
    matches: list[tuple[str, float]] = []
    i = 0
    while i < len(tokens):
        # longest entry first: three-word, two-word ("se desploma", "desempleo cae"), single word
        for width in (3, 2, 1):
            word = " ".join(tokens[i:i + width])
            if i + width <= len(tokens) and word in lexicon:
                valence = lexicon[word]
                break
        else:
            i += 1
            continue

        window = tokens[max(0, i - NEGATION_WINDOW):i]
        if i > 0 and tokens[i - 1] in boosters:
            valence += math.copysign(boosters[tokens[i - 1]], valence)
        if any(t in negators for t in window):
            valence *= NEGATION_FACTOR
        total += valence
        matches.append((word, round(valence, 2)))
        i += width

    compound = total / math.sqrt(total * total + ALPHA) if total else 0.0
    return {"compound": round(compound, 4), "matches": matches}
