"""Train the symptom -> disease classifier and save artifacts.

Run:  python -m training.train
Key choices (see README trade-offs):
  * dedupe rows BEFORE splitting (the raw CSV is ~94% duplicates -> leakage otherwise)
  * augment training rows with random symptom subsets, because real users type 2-4 symptoms
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.preprocessing import SymptomPreprocessor, normalize_token

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
SEED = 42
N_AUG = 12  # random subsets per training row


def load_rows() -> tuple[list[str], list[list[str]]]:
    df = pd.read_csv(DATA / "dataset.csv")
    diseases, rows = [], []
    for _, r in df.iterrows():
        syms = sorted({normalize_token(s) for s in r.iloc[1:] if isinstance(s, str) and s.strip()})
        diseases.append(r.iloc[0].strip())
        rows.append(syms)
    # dedupe on (disease, symptom set)
    seen, d2, r2 = set(), [], []
    for d, s in zip(diseases, rows):
        key = (d, tuple(s))
        if key not in seen:
            seen.add(key)
            d2.append(d)
            r2.append(s)
    return d2, r2


def augment(rows, labels, rng, n_aug):
    out_rows, out_labels = list(rows), list(labels)
    for s, y in zip(rows, labels):
        for _ in range(n_aug):
            k = int(rng.integers(1, max(2, len(s)) + 1))
            out_rows.append(sorted(rng.choice(s, size=min(k, len(s)), replace=False).tolist()))
            out_labels.append(y)
    return out_rows, out_labels


def topk_acc(model, X, y, k):
    proba = model.predict_proba(X)
    top = np.argsort(-proba, axis=1)[:, :k]
    classes = model.classes_
    return float(np.mean([y_i in classes[t] for y_i, t in zip(y, top)]))


def main():
    labels, rows = load_rows()
    vocab = sorted({s for r in rows for s in r})
    pre = SymptomPreprocessor(vocab)
    print(f"unique rows={len(rows)} diseases={len(set(labels))} symptoms={len(vocab)}")

    idx = np.arange(len(rows))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
    rng = np.random.default_rng(SEED)

    tr_rows, tr_y = augment([rows[i] for i in tr], [labels[i] for i in tr], rng, N_AUG)
    X_tr = np.stack([pre.vectorize(r) for r in tr_rows])
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    model.fit(X_tr, tr_y)

    te_rows, te_y = [rows[i] for i in te], [labels[i] for i in te]
    X_full = np.stack([pre.vectorize(r) for r in te_rows])
    # harder, realistic eval: user reports only 3 of their symptoms
    sub_rows = [sorted(rng.choice(s, size=min(3, len(s)), replace=False).tolist()) for s in te_rows]
    X_sub = np.stack([pre.vectorize(r) for r in sub_rows])
    metrics = {
        "n_unique_rows": len(rows),
        "n_train_rows_after_augmentation": len(tr_rows),
        "n_test_rows": len(te_rows),
        "top1_full_symptoms": round(topk_acc(model, X_full, te_y, 1), 3),
        "top3_full_symptoms": round(topk_acc(model, X_full, te_y, 3), 3),
        "top1_3_symptoms_only": round(topk_acc(model, X_sub, te_y, 1), 3),
        "top3_3_symptoms_only": round(topk_acc(model, X_sub, te_y, 3), 3),
    }
    print(json.dumps(metrics, indent=2))

    # Final model: refit on all data so shipped model sees every disease pattern.
    all_rows, all_y = augment(rows, labels, rng, N_AUG)
    model.fit(np.stack([pre.vectorize(r) for r in all_rows]), all_y)

    ART.mkdir(exist_ok=True)
    joblib.dump(model, ART / "model.joblib", compress=3)
    (ART / "vocabulary.json").write_text(json.dumps(vocab, indent=1))
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("saved artifacts to", ART)


if __name__ == "__main__":
    main()
