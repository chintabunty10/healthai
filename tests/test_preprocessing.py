import numpy as np
import pytest

from app.preprocessing import SymptomPreprocessor, normalize_token, parse_symptoms

VOCAB = ["abdominal_pain", "cough", "high_fever", "vomiting", "skin_rash", "headache", "fatigue"]


@pytest.fixture
def pre():
    return SymptomPreprocessor(VOCAB)


def test_normalize_token_cleans_case_space_punctuation():
    assert normalize_token("  Skin Rash!! ") == "skin_rash"
    assert normalize_token("dischromic _patches") == "dischromic_patches"


def test_parse_splits_commas_and_conjunctions():
    assert parse_symptoms("Fever, cough and  headache") == ["fever", "cough", "headache"]
    assert parse_symptoms(["Fever; Cough", "  "]) == ["fever", "cough"]


def test_exact_alias_and_fuzzy_resolution(pre):
    assert pre.resolve("Cough") == "cough"
    assert pre.resolve("tummy ache") == "abdominal_pain"
    assert pre.resolve("fevar") == "high_fever"  # typo


def test_unknown_symptom_reported_not_dropped_silently(pre):
    r = pre.process("cough, sparkly elbows")
    assert r.matched == ["cough"]
    assert r.unmatched == ["sparkly elbows"]


def test_output_is_standardized_vector(pre):
    r = pre.process("Vomiting, COUGH")
    assert r.vector.shape == (len(VOCAB),)
    assert r.vector.dtype == np.float32
    assert r.vector.sum() == 2
    assert r.vector[VOCAB.index("cough")] == 1


def test_cache_key_is_order_and_case_insensitive(pre):
    a = pre.process("Fever, cough").cache_key
    b = pre.process("cough ,FEVER, fever").cache_key
    assert a == b


def test_validation_limits(pre):
    with pytest.raises(ValueError):
        pre.process(",".join(["cough"] * 25) + ",a1,a2,a3,a4,a5,a6")  # >20 distinct phrases after parse
    with pytest.raises(ValueError):
        pre.process("x" * 100)
    assert pre.process("   ").matched == []
