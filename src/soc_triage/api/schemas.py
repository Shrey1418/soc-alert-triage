from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    OrgId: int
    IncidentId: int
    AlertId: int
    DetectorId: int
    AlertTitle: int
    Timestamp: str
    MitreTechniques: str | None = None


class PredictRequest(BaseModel):
    evidence: list[Evidence] = Field(min_length=1, max_length=5000)


class IncidentResult(BaseModel):
    OrgId: int
    IncidentId: int
    decision: str
    action: str
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    results: list[IncidentResult]
