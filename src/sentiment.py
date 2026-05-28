"""Sentiment analysis via VADER (offline) with optional Claude API enhancement."""
import os
import json
from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

TOPIC_KEYWORDS = {
    "Politics":    ["election", "president", "government", "minister", "congress", "senate", "vote", "policy"],
    "Economy":     ["inflation", "gdp", "market", "stock", "trade", "tariff", "recession", "growth", "fed", "bank"],
    "Technology":  ["ai", "artificial intelligence", "tech", "software", "data", "cyber", "digital", "startup"],
    "Conflict":    ["war", "military", "attack", "conflict", "troops", "ceasefire", "missile", "bomb"],
    "Climate":     ["climate", "carbon", "emission", "temperature", "flood", "wildfire", "renewable", "energy"],
    "Health":      ["health", "hospital", "vaccine", "disease", "pandemic", "drug", "cancer", "medicine"],
    "Business":    ["merger", "acquisition", "revenue", "profit", "ceo", "company", "earnings", "deal"],
}

def classify_topic(text: str) -> str:
    text_lower = text.lower()
    scores = {topic: sum(kw in text_lower for kw in keywords)
              for topic, keywords in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"

def vader_sentiment(text: str) -> dict:
    scores = _vader.polarity_scores(text)
    label = "positive" if scores["compound"] >= 0.05 else \
            "negative" if scores["compound"] <= -0.05 else "neutral"
    return {"compound": scores["compound"], "label": label,
            "pos": scores["pos"], "neg": scores["neg"], "neu": scores["neu"]}

def analyze_batch(articles) -> list[dict]:
    results = []
    for art in articles:
        sentiment = vader_sentiment(art.full_text)
        topic     = classify_topic(art.full_text)
        results.append({
            "source":    art.source,
            "title":     art.title,
            "topic":     topic,
            "sentiment": sentiment["label"],
            "score":     sentiment["compound"],
            "pos":       sentiment["pos"],
            "neg":       sentiment["neg"],
            "neu":       sentiment["neu"],
            "link":      art.link,
        })
    return results
