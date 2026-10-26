from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl


class Intent(str, Enum):
    SYMPTOMS = "symptoms"
    MEDICINE = "medicine"
    VISIT = "visit"
    REPORT = "report"
    GENERAL = "general"
    URGENT = "urgent"


class Risk(BaseModel):
    level: Literal["ROUTINE", "CAUTION", "CRITICAL"] = "ROUTINE"
    reasons: list[str] = Field(default_factory=list)
    rule_version: str = "screening-1"


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    user: str = Field(default="demo", min_length=1, max_length=100)
    session: str = Field(default="", max_length=100)
    subject: str = Field(default="self", min_length=1, max_length=40)
    context: Literal["selective", "full"] = "selective"


class Understanding(BaseModel):
    primary: Intent = Intent.GENERAL
    scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=0, ge=0, le=1)
    signals: dict[str, dict[str, float]] = Field(default_factory=dict)


class Reference(BaseModel):
    key: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    domain: Intent
    text: str = Field(min_length=1, max_length=20000)
    url: HttpUrl
    version: str = Field(min_length=1, max_length=100)
    scope: str = Field(min_length=1, max_length=300)
    approved: bool = False
    entities: list[str] = Field(default_factory=list, max_length=20)


class Answer(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    questions: list[str] = Field(default_factory=list, max_length=6)
    next_steps: list[str] = Field(default_factory=list, max_length=6)
    limitations: list[str] = Field(default_factory=list, max_length=6)
    reference_keys: list[str] = Field(default_factory=list, max_length=8)


class Reply(BaseModel):
    request_id: str
    session: str
    intent: Understanding
    risk: Risk
    primary_agent: str
    supporting_agents: list[str] = Field(default_factory=list)
    answer: Answer
    references: list[Reference] = Field(default_factory=list)
    degraded: bool = False
    storage: Literal["available", "degraded", "skipped", "memory_preview"]
    usage: dict
    elapsed_ms: float
    handoff_connected: bool = False


class DocumentBatch(BaseModel):
    documents: list[Reference] = Field(min_length=1, max_length=50)
