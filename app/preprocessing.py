"""Input cleaning + normalization: raw user text -> canonical symptoms -> model vector.

Pipeline: parse -> validate -> normalize tokens -> alias/fuzzy match -> multi-hot vector.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import numpy as np

MAX_SYMPTOMS = 20
MAX_SYMPTOM_LEN = 60
FUZZY_CUTOFF = 0.8

# Everyday phrasing -> canonical dataset symptom.
ALIASES: dict[str, str] = {
    "tummy ache": "abdominal_pain",
    "stomach ache": "abdominal_pain",
    "stomach pain": "abdominal_pain",
    "belly pain": "abdominal_pain",
    "throwing up": "vomiting",
    "puking": "vomiting",
    "high temperature": "high_fever",
    "fever": "high_fever",
    "temperature": "high_fever",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "tiredness": "fatigue",
    "runny nose": "runny_nose",
    "stuffy nose": "congestion",
    "blocked nose": "congestion",
    "sore throat": "throat_irritation",
    "headache": "headache",
    "dizzy": "dizziness",
    "loose motion": "diarrhoea",
    "diarrhea": "diarrhoea",
    "itchy": "itching",
    "rash": "skin_rash",
    "cant sleep": "restlessness",
    "short of breath": "breathlessness",
    "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "body ache": "muscle_pain",
    "body pain": "muscle_pain",
    "joint ache": "joint_pain",
    "pee burning": "burning_micturition",
    "burning urination": "burning_micturition",
    "yellow skin": "yellowish_skin",
    "weight loss": "weight_loss",
}

_SPLIT_RE = re.compile(r"[,;\n]+|\band\b|\bwith\b|&")


def normalize_token(text: str) -> str:
    """lowercase, drop punctuation, collapse whitespace/underscores -> 'snake_case'."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_symptoms(raw: str | list[str]) -> list[str]:
    """Accept 'fever, cough and chills' or ['fever', 'cough'] -> list of cleaned phrases."""
    if isinstance(raw, str):
        parts = _SPLIT_RE.split(raw.lower())
    else:
        parts = []
        for item in raw:
            parts.extend(_SPLIT_RE.split(str(item).lower()))
    cleaned = [re.sub(r"\s+", " ", p).strip() for p in parts]
    return [p for p in cleaned if p]


@dataclass
class PreprocessResult:
    matched: list[str] = field(default_factory=list)      # canonical symptoms, sorted, unique
    unmatched: list[str] = field(default_factory=list)    # phrases we could not map
    vector: np.ndarray | None = None

    @property
    def cache_key(self) -> str:
        return "|".join(self.matched)


class SymptomPreprocessor:
    def __init__(self, vocabulary: list[str]):
        self.vocabulary = list(vocabulary)
        self._index = {s: i for i, s in enumerate(self.vocabulary)}
        self._alias = {normalize_token(k): v for k, v in ALIASES.items() if v in self._index}

    def resolve(self, phrase: str) -> str | None:
        """Map one cleaned phrase to a canonical symptom (exact, alias, then fuzzy)."""
        token = normalize_token(phrase)
        if not token:
            return None
        if token in self._index:
            return token
        if token in self._alias:
            return self._alias[token]
        candidates = list(self._index) + list(self._alias)
        close = difflib.get_close_matches(token, candidates, n=1, cutoff=FUZZY_CUTOFF)
        if close:
            hit = close[0]
            return hit if hit in self._index else self._alias[hit]
        return None

    def process(self, raw: str | list[str]) -> PreprocessResult:
        phrases = parse_symptoms(raw)
        if len(phrases) > MAX_SYMPTOMS:
            raise ValueError(f"Too many symptoms (max {MAX_SYMPTOMS}).")
        if any(len(p) > MAX_SYMPTOM_LEN for p in phrases):
            raise ValueError(f"Each symptom must be at most {MAX_SYMPTOM_LEN} characters.")
        matched: set[str] = set()
        unmatched: list[str] = []
        for p in phrases:
            hit = self.resolve(p)
            if hit:
                matched.add(hit)
            elif p not in unmatched:
                unmatched.append(p)
        result = PreprocessResult(matched=sorted(matched), unmatched=unmatched)
        result.vector = self.vectorize(result.matched)
        return result

    def vectorize(self, symptoms: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        for s in symptoms:
            vec[self._index[s]] = 1.0
        return vec
