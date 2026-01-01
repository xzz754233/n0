import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from src.configuration import Configuration
from src.llm_service import create_llm_structured_model, create_llm_with_tools
from src.prompts import (
    events_summarizer_prompt,
    lead_researcher_prompt,
    structure_events_prompt,
)
from src.research_events.research_events_graph import research_events_app
from src.research_events.merge_events.utils import ensure_categories_with_events
from src.state import (
    CategoriesWithEvents,
    Chronology,
    FinishResearchTool,
    ResearchEventsTool,
    SupervisorState,
    SupervisorStateInput,
)
from src.utils import get_buffer_string_with_tools, get_langfuse_handler, think_tool

config = Configuration()
MAX_TOOL_CALL_ITERATIONS = config.max_tool_iterations


async def supervisor_node(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor_tools"]]:
    """The 'brain' of the agent."""
    tools = [ResearchEventsTool, FinishResearchTool, think_tool]
    tools_model = create_llm_with_tools(tools=tools, config=config)

    messages = state.get("conversation_history", [])
    last_message = messages[-1] if messages else ""

    system_message = SystemMessage(
        content=lead_researcher_prompt.format(
            person_to_research=state["person_to_research"],
            events_summary=state.get("events_summary", "Everything is missing"),
            last_message=last_message,
        )
    )

    human_message = HumanMessage(content="Start the research process.")
    prompt = [system_message, human_message]

    # Invoke model
    response = await tools_model.ainvoke(prompt)

    return Command(
        goto="supervisor_tools",
        update={
            "conversation_history": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        },
    )


async def supervisor_tools_node(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor", "structure_events"]]:
    """The 'hands' of the agent."""

    # Safe State Loading
    raw_events = state.get("existing_events")
    if not raw_events:
        existing_events = CategoriesWithEvents(
            context="", conflict="", reaction="", outcome=""
        )
    else:
        existing_events = ensure_categories_with_events(raw_events)

    events_summary = state.get("events_summary", "")
    used_domains = state.get("used_domains", [])
    last_message = state["conversation_history"][-1]
    iteration_count = state.get("iteration_count", 0)

    # --- CRITICAL FIX: HANDLE EMPTY TOOL CALLS (CHATTER) ---
    if not last_message.tool_calls:
        if iteration_count >= MAX_TOOL_CALL_ITERATIONS:
            return Command(goto="structure_events")

        # If model just talked without calling a tool, FORCE it to retry
        print("⚠️ Model chattered without tool call. Forcing retry.")
        return Command(
            goto="supervisor",
            update={
                "conversation_history": [
                    HumanMessage(
                        content="ERROR: You did not call a tool. You MUST call a tool (think_tool or ResearchEventsTool) to proceed. Do not output text."
                    )
                ]
            },
        )
    # -------------------------------------------------------

    all_tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # FIX: Handle JSON string args
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except:
                tool_args = {}

        if tool_name == "FinishResearchTool":
            return Command(goto="structure_events")

        elif tool_name == "think_tool":
            response_content = tool_args.get("reflection", "Reflection recorded.")
            all_tool_messages.append(
                ToolMessage(
                    content=response_content,
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

        elif tool_name == "ResearchEventsTool":
            research_question = tool_args.get("research_question", "")

            result = await research_events_app.ainvoke(
                {
                    "research_question": research_question,
                    "existing_events": existing_events,
                    "used_domains": used_domains,
                }
            )

            existing_events = ensure_categories_with_events(result["existing_events"])
            used_domains = result["used_domains"]

            summarizer_prompt = events_summarizer_prompt.format(
                existing_events=existing_events
            )
            response = await create_llm_structured_model(config=config).ainvoke(
                summarizer_prompt
            )
            events_summary = response.content

            all_tool_messages.append(
                ToolMessage(
                    content=f"Research complete for: {research_question}",
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

    return Command(
        goto="supervisor",
        update={
            "existing_events": existing_events,
            "conversation_history": all_tool_messages,
            "used_domains": used_domains,
            "events_summary": events_summary,
        },
    )


async def structure_events(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """Step 2: Structures events into JSON + Clean up."""
    print("--- Step 2: Structuring Events into JSON ---")

    raw_events = state.get("existing_events")
    if not raw_events:
        return {"structured_events": []}

    existing_events = ensure_categories_with_events(raw_events)
    structured_llm = create_llm_structured_model(config=config, class_name=Chronology)

    all_events = []
    # Sequential processing for stability
    for category in ["context", "conflict", "reaction", "outcome"]:
        text = getattr(existing_events, category, "")
        if text:
            try:
                resp = await structured_llm.ainvoke(
                    structure_events_prompt.format(existing_events=text)
                )
                all_events.extend(resp.events)
            except Exception as e:
                print(f"Error structuring {category}: {e}")

    # --- FINAL CLEANUP ---
    cleaned_events = []
    for event in all_events:
        # 1. Fix Broken Text
        if event.description:
            event.description = event.description.strip().strip("\\").strip('"').strip()
            if not event.description.endswith((".", "!", "?", '"')):
                event.description += "."

        if event.name:
            event.name = event.name.strip().strip("\\").strip('"').strip()

        # 2. Fix Unknown Locations
        if not event.location or event.location.lower() in ["none", "unknown", "null"]:
            event.location = "Internet / General"

        cleaned_events.append(event)

    return {"structured_events": cleaned_events}


workflow = StateGraph(SupervisorState, input_schema=SupervisorStateInput)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("supervisor_tools", supervisor_tools_node)
workflow.add_node("structure_events", structure_events)
workflow.add_edge(START, "supervisor")

graph = workflow.compile().with_config({"callbacks": [get_langfuse_handler()]})
