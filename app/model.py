"""Model service: loads the trained classifier + disease knowledge, returns ranked conditions."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocessing import SymptomPreprocessor
from .schemas import Condition

ROOT = Path(__file__).resolve().parent.parent
RED_FLAGS = {"chest_pain", "breathlessness", "altered_sensorium", "coma", "slurred_speech",
             "weakness_of_one_body_side", "blood_in_sputum", "bloody_stool", "stomach_bleeding"}
URGENT_MESSAGE = ("One or more of your symptoms can indicate a medical emergency. "
                  "Please seek urgent medical care or call your local emergency number.")
DISCLAIMER = ("This tool gives general information only, is not a diagnosis, and does not replace "
              "advice from a qualified healthcare professional.")


NAME_FIXES = {
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids (piles)",
    "(vertigo) Paroymsal Positional Vertigo": "Paroxysmal Positional Vertigo (vertigo)",
    "Osteoarthristis": "Osteoarthritis",
    "Dimorphic hemorrhoids(piles)": "Dimorphic hemorrhoids (piles)",
}


def clean_name(name: str) -> str:
    """Fix typos / stray whitespace in the source dataset's disease labels for display."""
    name = " ".join(str(name).split())
    return NAME_FIXES.get(name, name)


class SymptomModel:
    def __init__(self, artifacts: Path = ROOT / "artifacts", data: Path = ROOT / "data"):
        self.model = joblib.load(artifacts / "model.joblib")
        self.model.n_jobs = 1  # single-row inference: thread-pool overhead hurts under concurrent load
        vocab = json.loads((artifacts / "vocabulary.json").read_text())
        self.preprocessor = SymptomPreprocessor(vocab)
        desc = pd.read_csv(data / "symptom_Description.csv")
        prec = pd.read_csv(data / "symptom_precaution.csv")
        self._desc = {clean_name(r.Disease): str(r.Description).strip() for r in desc.itertuples()}
        self._prec = {
            clean_name(r[0]): [str(x).strip().capitalize() for x in r[1:] if isinstance(x, str) and x.strip()]
            for r in prec.itertuples(index=False)
        }

    @staticmethod
    def urgent(symptoms: list[str]) -> str | None:
        return URGENT_MESSAGE if RED_FLAGS.intersection(symptoms) else None

    def predict(self, vector: np.ndarray, top_k: int = 5, min_probability: float = 0.03) -> list[Condition]:
        proba = self.model.predict_proba(vector.reshape(1, -1))[0]
        order = np.argsort(-proba)[:top_k]
        out = []
        for i in order:
            p = float(proba[i])
            if p < min_probability and out:
                break
            name = clean_name(self.model.classes_[i])
            out.append(Condition(name=name, probability=round(p, 3),
                                 description=self._desc.get(name, ""),
                                 precautions=self._prec.get(name, [])))
        return out
