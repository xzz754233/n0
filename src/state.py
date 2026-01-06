import operator
import uuid
from typing import Annotated, List, TypedDict, Optional
from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel, Field, field_validator


# --- 共用的文字清理邏輯 ---
def clean_string_field(cls, v):
    if not v:
        return v
    v = str(v).strip()
    if v.endswith("\\"):
        v = v[:-1] + "'s"
    if v.count('"') % 2 != 0:
        v += '"'
    v = v.replace("\\'", "'")
    return v


# [NEW] 證據單元：你的產品核心
class EvidencePoint(BaseModel):
    """A specific piece of evidence supporting or debunking the claim."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])

    # 標題，例如 "2023 Meta-Analysis" 或 "Reaction Time Data"
    topic: str = Field(description="Short headline for the evidence point.")

    # 詳細內容，包含數據、P值、樣本數
    details: str = Field(
        description="The key finding. Specific numbers and study details."
    )

    # 立場：這條證據是支持還是反對？
    stance: str = Field(
        description="Does this support the claim? 'Supports', 'Debunks', or 'Nuanced'."
    )

    # [CRITICAL] 來源引用
    source_title: str = Field(
        default="Source",
        description="Short title for the source (e.g. 'Harvard Health', 'Nature Journal').",
    )
    source_url: str = Field(default="", description="The original URL.")

    # 方便前端直接使用的 Markdown 連結
    @property
    def formatted_citation(self) -> str:
        if self.source_url:
            return f"[{self.source_title}]({self.source_url})"
        return self.source_title

    @field_validator("details", "topic", "source_title", mode="before")
    @classmethod
    def clean_text(cls, v):
        return clean_string_field(cls, v)


# [NEW] 用於 Merge 階段的分類容器
class CategoriesWithEvents(BaseModel):
    """Structure for categorizing myth-busting findings."""

    origin_of_belief: str = Field(
        default="",
        description="Why do people believe this? Cultural origins, old wisdom, or viral misunderstandings.",
    )
    scientific_evidence: str = Field(
        default="",
        description="Hard Data. Peer-reviewed studies, meta-analyses, statistics, sample sizes.",
    )
    expert_consensus: str = Field(
        default="",
        description="What major organizations (WHO, NASA, FDA) or top experts agree on.",
    )
    final_verdict: str = Field(
        default="",
        description="The Conclusion. BUSTED, CONFIRMED, or PLAUSIBLE. Including the nuance.",
    )


# [UPDATED] 中間產物：原始提取事件
class RawEvent(BaseModel):
    """Represents a raw finding extracted from a text chunk."""

    description: str = Field(description="The finding details.")
    # 在 Fact Check 中，這裡可以用來標註 "Study Year" 或 "Credibility"
    date_context: str | None = Field(
        None, description="Context (Year of study, type of source)."
    )
    category: str = Field(description="Category: origin, science, consensus, verdict.")
    source_url: str = Field(default="", description="Where this event came from.")

    @field_validator("description", "category", "date_context", mode="before")
    @classmethod
    def clean_text(cls, v):
        return clean_string_field(cls, v)


# 為了兼容 LLM 結構化輸出
class FactCheckReport(BaseModel):
    evidence_points: list[EvidencePoint]


# --- Tools ---
class ResearchEventsTool(BaseModel):
    research_question: str


class FinishResearchTool(BaseModel):
    pass


# --- State ---
def override_reducer(current_value, new_value):
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


class ResearchState(TypedDict):
    research_question: str
    target_urls: List[str]
    processed_urls: List[str]
    gathered_events: Annotated[List[RawEvent], operator.add]
    final_evidence: List[EvidencePoint]  # Output


class SupervisorStateInput(TypedDict):
    person_to_research: str  # Input Claim


class SupervisorState(TypedDict):
    person_to_research: str
    conversation_history: Annotated[list[MessageLikeRepresentation], override_reducer]
    iteration_count: int
    events_summary: str

    # [UPDATED] 最終結果存這裡
    evidence_points: List[EvidencePoint]
