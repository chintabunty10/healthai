"""LLM explanations via Groq's OpenAI-compatible API, with a safe non-LLM fallback."""
from __future__ import annotations

import logging

log = logging.getLogger("healthai.llm")

SYSTEM_PROMPT = (
    "You are a careful health-information assistant for a symptom-checker app. "
    "Explain, in plain language and under 120 words, what the given condition is, why the "
    "user's reported symptoms could relate to it, and when to see a doctor. "
    "Never give a diagnosis, never recommend specific drugs or doses, and end by advising the "
    "user to consult a healthcare professional."
)


class Explainer:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.client = None
        if api_key:
            from openai import OpenAI  # lazy import
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=15.0, max_retries=1)

    def explain(self, condition: str, symptoms: list[str], fallback_text: str = "") -> tuple[str, str]:
        """Returns (text, source) where source is 'llm' or 'fallback'."""
        if self.client:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Condition: {condition}\nReported symptoms: {', '.join(symptoms)}")},
                    ],
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text, "llm"
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM call failed: %s", type(exc).__name__)
        return self.fallback(condition, fallback_text), "fallback"

    @staticmethod
    def fallback(condition: str, fallback_text: str = "") -> str:
        """Non-LLM answer used when the provider is down, slow or saturated."""
        base = fallback_text or f"{condition} is a condition that can match the symptoms you reported."
        return f"{base} Please consult a healthcare professional for a proper evaluation."
