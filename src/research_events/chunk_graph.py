from typing import Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from src.configuration import Configuration
from src.llm_service import create_llm_chunk_model


# CHANGED: Renamed from BiographicEventCheck to match our new domain
class DramaEventCheck(BaseModel):
    contains_drama_event: bool = Field(
        description="Whether the text chunk contains relevant drama, scandal, or conflict information"
    )


class ChunkResult(BaseModel):
    content: str
    # CHANGED: Updated field name
    contains_drama_event: bool = Field(
        description="Whether the text chunk contains relevant drama, scandal, or conflict information"
    )


class ChunkState(TypedDict):
    text: str
    chunks: List[str]
    results: Dict[str, ChunkResult]


def split_text(state: ChunkState) -> ChunkState:
    """Split text into smaller chunks."""
    text = state["text"]
    chunk_size = 2000
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return {"chunks": chunks}


def check_chunk_for_events(state: ChunkState, config) -> ChunkState:
    """Check each chunk for drama/scandal events using structured output."""
    # CHANGED: Using the new Pydantic model
    model = create_llm_chunk_model(config, DramaEventCheck)
    results = {}

    for i, chunk in enumerate(state["chunks"]):
        # CHANGED: Completely rewrote the prompt to detect "Tea" instead of "History"
        prompt = f"""
        Analyze this text chunk and determine if it contains SPECIFIC details regarding a scandal, controversy, or internet drama.
        
        ONLY mark as true if the chunk contains:
        - Specific accusations, allegations, or "call outs"
        - Details of a conflict (arguments, fights, breakups, beefs)
        - "Receipts" (descriptions of screenshots, leaked messages, photos, recordings)
        - Official statements, apologies, or press releases (e.g., "Notes App apology")
        - Significant social media actions (unfollowing, blocking, deleting posts, viral threads)
        - Legal actions (lawsuits, restraining orders, police reports)
        - Concrete dates/locations/platforms associated with the incident
        
        DO NOT mark as true for:
        - Website navigation menus, footers, or headers
        - Generic advertisements or spam
        - General biographical facts unrelated to the controversy (e.g., where they went to high school, unless relevant to the scandal)
        - General descriptions of their career that are not part of the drama
        
        The info must be specific "tea" or context for the drama, not just filler text.
        
        Text chunk: "{chunk}"
        """

        result = model.invoke(prompt)
        results[f"chunk_{i}"] = ChunkResult(
            content=chunk, contains_drama_event=result.contains_drama_event
        )

    return {"results": results}


# CHANGED: Renamed function to reflect purpose (Note: We must update the import in merge_events_graph.py)
def create_drama_event_graph() -> CompiledStateGraph:
    """Create and return the drama/scandal event detection graph."""
    graph = StateGraph(ChunkState, config_schema=Configuration)

    graph.add_node("split_text", split_text)
    graph.add_node("check_events", check_chunk_for_events)

    graph.add_edge(START, "split_text")
    graph.add_edge("split_text", "check_events")
    graph.add_edge("check_events", END)

    return graph.compile()


# Export the graph
graph = create_drama_event_graph()
