from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    symptoms: str = Field(..., min_length=1, max_length=500, description="e.g. 'fever, cough and chills'")

    @field_validator("symptoms")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("symptoms must not be blank")
        return v


class Condition(BaseModel):
    name: str
    probability: float
    description: str = ""
    precautions: list[str] = []


class PredictResponse(BaseModel):
    matched_symptoms: list[str]
    unmatched_symptoms: list[str]
    conditions: list[Condition]
    cached: bool
    disclaimer: str
    urgent_warning: str | None = None


class ExplainRequest(BaseModel):
    condition: str = Field(..., min_length=1, max_length=100)
    symptoms: list[str] = Field(..., min_length=1, max_length=20)


class ExplainResponse(BaseModel):
    condition: str
    explanation: str
    cached: bool
    source: str  # "llm" or "fallback"
