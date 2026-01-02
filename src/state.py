import operator
from typing import Annotated, List, TypedDict, Optional

from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel, Field

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


# 最終輸出的格式 (保持不變，確保前端/輸出邏輯不壞)
class ChronologyEvent(BaseModel):
    id: str = Field(description="Unique ID, e.g., 'event_001'")
    name: str = Field(description="Short headline.")
    description: str = Field(description="Full description.")
    date: ChronologyDate = Field(..., description="Date structure.")
    location: str = Field(default="Internet", description="Location.")
    source_url: str = Field(
        default="", description="Source URL for this event."
    )  # 新增：方便查證


class Chronology(BaseModel):
    events: list[ChronologyEvent]


# --- [NEW] 中間產物：原始提取事件 ---
# 這是為了解決 "文字切斷" 和 "Token浪費" 新增的結構
# 我們不再用長字串 (CategoriesWithEvents) 來傳遞資訊，改用這個物件列表
class RawEvent(BaseModel):
    """Represents a raw event extracted from a text chunk."""

    description: str = Field(description="The event details.")
    date_context: str = Field(description="Date mentioned in text (e.g., 'late 2024').")
    category: str = Field(description="Category: context, conflict, reaction, outcome.")
    source_url: str = Field(default="", description="Where this event came from.")


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


# --- [NEW] Research Subgraph State (Map-Reduce 架構) ---
class ResearchState(TypedDict):
    # Inputs
    research_question: str

    # Process Control
    target_urls: List[str]  # 待爬取的 URL 隊列
    processed_urls: List[str]  # 已爬過的 URL

    # Data Accumulation (核心優化：使用 operator.add 自動合併列表)
    # 這讓 LangGraph 自動把新爬到的 RawEvent append 到列表，而不是覆蓋
    gathered_events: Annotated[List[RawEvent], operator.add]

    # Outputs (Sub-graph 的輸出)
    final_timeline: List[ChronologyEvent]


# --- Main Supervisor Graph State ---


class SupervisorStateInput(TypedDict):
    """The initial input to start the main research graph."""

    person_to_research: str
    # 我們移除了 existing_events，因為不再需要外部傳入那個長字串結構


class SupervisorState(TypedDict):
    """The complete state for the main supervisor graph."""

    person_to_research: str
    conversation_history: Annotated[list[MessageLikeRepresentation], override_reducer]
    iteration_count: int

    # 用於 UI 顯示摘要 (可選)
    events_summary: str

    # 這裡存最終結果 (整合了原本的 final_events 和 structured_events)
    structured_events: List[ChronologyEvent]
