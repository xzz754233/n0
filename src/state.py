import operator
from typing import Annotated, List, TypedDict, Optional
import re

from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel, Field, field_validator

################################################################################
# Section 1: Data Models
################################################################################


class ChronologyDate(BaseModel):
    """A structured representation of a date."""

    year: int | None = Field(None, description="The year of the event.")
    note: str | None = Field(
        None,
        description="Specifics (month, day, specific time). Ex: 'May 5th'",
    )


# --- 共用的文字清理邏輯 ---
def clean_string_field(cls, v):
    if not v:
        return v
    v = str(v).strip()

    # 1. 修復結尾的轉義斜線 (Willy\ -> Willy's)
    if v.endswith("\\"):
        v = v[:-1] + "'s"  # 猜測這通常是所有格被切斷

    # 2. 修復被截斷的引號 (The "Willy -> The "Willy")
    # 計算引號數量，如果是奇數，補一個
    if v.count('"') % 2 != 0:
        v += '"'

    # 3. 移除不必要的開頭轉義
    v = v.replace("\\'", "'")

    return v


# 最終輸出的格式
class ChronologyEvent(BaseModel):
    id: str = Field(description="Unique ID, e.g., 'event_001'")
    name: str = Field(description="Short headline.")
    description: str = Field(description="Full description.")
    date: ChronologyDate = Field(..., description="Date structure.")
    location: str = Field(default="Internet", description="Location.")
    source_url: str = Field(default="", description="Source URL for this event.")

    # 加入驗證器
    @field_validator("name", "description", mode="before")
    @classmethod
    def clean_text(cls, v):
        return clean_string_field(cls, v)


class Chronology(BaseModel):
    events: list[ChronologyEvent]


# --- [NEW] 中間產物：原始提取事件 ---
class RawEvent(BaseModel):
    """Represents a raw event extracted from a text chunk."""

    description: str = Field(description="The event details.")
    date_context: str = Field(description="Date mentioned in text (e.g., 'late 2024').")
    category: str = Field(description="Category: context, conflict, reaction, outcome.")
    source_url: str = Field(default="", description="Where this event came from.")

    # 加入驗證器：在提取階段就修復，這是最關鍵的一步
    @field_validator("description", "category", "date_context", mode="before")
    @classmethod
    def clean_text(cls, v):
        return clean_string_field(cls, v)


################################################################################
# Section 2: Agent Tools
################################################################################


class ResearchEventsTool(BaseModel):
    research_question: str


class FinishResearchTool(BaseModel):
    pass


################################################################################
# Section 3: Graph State Definitions
################################################################################


def override_reducer(current_value, new_value):
    """Allows replacing the value completely."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


class ResearchState(TypedDict):
    # Inputs
    research_question: str

    # Process Control
    target_urls: List[str]  # 待爬取的 URL 隊列
    processed_urls: List[str]  # 已爬過的 URL

    # Data Accumulation
    gathered_events: Annotated[List[RawEvent], operator.add]

    # Outputs (Sub-graph 的輸出)
    final_timeline: List[ChronologyEvent]


class SupervisorStateInput(TypedDict):
    """The initial input to start the main research graph."""

    person_to_research: str


class SupervisorState(TypedDict):
    """The complete state for the main supervisor graph."""

    person_to_research: str
    conversation_history: Annotated[list[MessageLikeRepresentation], override_reducer]
    iteration_count: int

    # 用於 UI 顯示摘要 (可選)
    events_summary: str

    # 這裡存最終結果
    structured_events: List[ChronologyEvent]
