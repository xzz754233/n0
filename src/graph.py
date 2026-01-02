# src/graph.py
import json
import uuid
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command

from src.configuration import Configuration
from src.llm_service import create_llm_with_tools, create_llm_structured_model
from src.prompts import (
    lead_researcher_prompt,
    structure_events_prompt,  # 這裡我們會稍微變通使用這個 prompt
)

# 引入新的子圖
from src.research_events.research_events_graph import research_events_app
from src.state import (
    Chronology,
    ChronologyEvent,
    ChronologyDate,
    FinishResearchTool,
    ResearchEventsTool,
    SupervisorState,
    SupervisorStateInput,
    RawEvent,
)
from src.utils import get_langfuse_handler, think_tool

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

    # 簡單的事件計數摘要，讓 Agent 知道進度
    current_events = state.get("structured_events", [])
    events_summary_text = f"Currently have {len(current_events)} raw events collected."
    if current_events:
        # 提供最近收集到的幾個事件標題，讓 Agent 不要鬼打牆
        titles = [e.name for e in current_events[-5:]]
        events_summary_text += f" Recent findings: {', '.join(titles)}"

    system_message = SystemMessage(
        content=lead_researcher_prompt.format(
            person_to_research=state["person_to_research"],
            events_summary=events_summary_text,
            last_message=last_message,
        )
    )

    human_message = HumanMessage(content="Start the research process.")
    prompt = [system_message, human_message]

    # 調用 LLM
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
    last_message = state["conversation_history"][-1]
    iteration_count = state.get("iteration_count", 0)

    # 強制結束條件
    if (
        not hasattr(last_message, "tool_calls")
        or not last_message.tool_calls
        or iteration_count >= MAX_TOOL_CALL_ITERATIONS
    ):
        return Command(goto="structure_events")

    all_tool_messages = []

    # 用於累加這一輪新發現的事件
    newly_found_chronology_events = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_id = tool_call.get("id")

        # JSON 解析保護
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
                    content=response_content, tool_call_id=tool_id, name=tool_name
                )
            )

        elif tool_name == "ResearchEventsTool":
            research_question = tool_args.get("research_question", "")
            print(f"🔍 Researching: {research_question}")

            try:
                # 調用子圖
                result = await research_events_app.ainvoke(
                    {
                        "research_question": research_question,
                        "target_urls": [],  # 初始化
                        "processed_urls": [],  # 可以考慮從 global state 傳入以全域去重
                        "gathered_events": [],  # 初始化
                    }
                )

                raw_events = result.get("gathered_events", [])

                # [Data Transformation] RawEvent -> ChronologyEvent
                # 將 RawEvent 轉為 ChronologyEvent 以便存入 Main State
                # 這一步是為了適配 SupervisorState 的型別定義
                for raw in raw_events:
                    # 嘗試從 date_context 提取年份，失敗則 None
                    year = None
                    import re

                    match = re.search(r"\d{4}", raw.date_context or "")
                    if match:
                        year = int(match.group(0))

                    newly_found_chronology_events.append(
                        ChronologyEvent(
                            id=str(uuid.uuid4())[:8],
                            name=f"Event from {raw.source_url[:30]}...",  # 暫時名稱
                            description=raw.description,
                            date=ChronologyDate(year=year, note=raw.date_context),
                            location="Internet",
                            source_url=raw.source_url,
                        )
                    )

                content_msg = (
                    f"Found {len(raw_events)} events related to {research_question}."
                )

            except Exception as e:
                print(f"❌ Error in ResearchEventsTool: {e}")
                content_msg = f"Error executing research: {str(e)}"

            all_tool_messages.append(
                ToolMessage(content=content_msg, tool_call_id=tool_id, name=tool_name)
            )

    # 更新 State：將新發現的事件加入 structured_events (利用 list concat)
    existing = state.get("structured_events", [])
    # 簡單過濾 None
    if existing is None:
        existing = []

    updated_events = existing + newly_found_chronology_events

    return Command(
        goto="supervisor",
        update={
            "conversation_history": all_tool_messages,
            "structured_events": updated_events,  # 這裡進行了 State 更新
        },
    )


async def structure_events(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """
    Step 3: Final Consolidation (The Reduce Step).
    Takes all raw/loose events and merges them into a clean timeline.
    """
    print("--- Final Step: Consolidating & Deduplicating Events ---")

    all_events = state.get("structured_events", [])
    if not all_events:
        return {"structured_events": []}

    # 1. 準備輸入資料：將大量事件轉為文字，供 LLM 整理
    # 如果事件非常多，可以考慮先按年份簡單排序，或分批處理
    # 這裡示範一次性處理 (Gemini Flash Context Window 很大)
    events_text_blob = ""
    for idx, e in enumerate(all_events):
        events_text_blob += f"Event {idx + 1}: [{e.date.note or e.date.year}] {e.description} (Source: {e.source_url})\n"

    # 2. 定義整理用的 Prompt
    consolidation_prompt = f"""
    You are a Timeline Editor. I have collected {len(all_events)} raw events.
    Many are duplicates or fragmented.
    
    Task:
    1. **Deduplicate**: Merge events that describe the same incident.
    2. **Chronological Order**: Sort strictly by date.
    3. **Fix Text**: Ensure descriptions are complete sentences.
    4. **Source**: Keep the most authoritative source URL.
    
    Input Events:
    {events_text_blob[:50000]}  # 簡單截斷防止溢出，雖然 Flash 可以吃 1M
    
    Return a clean JSON list of events.
    """

    # 3. 調用 LLM 生成最終 JSON
    structured_llm = create_llm_structured_model(config=config, class_name=Chronology)

    try:
        final_result = await structured_llm.ainvoke(consolidation_prompt)
        final_events = final_result.events

        # 簡單後處理
        for e in final_events:
            if not e.id:
                e.id = str(uuid.uuid4())[:8]

        print(f"✅ Final timeline generated with {len(final_events)} events.")

    except Exception as e:
        print(f"❌ Error in final consolidation: {e}")
        # 如果失敗，回傳原始列表（至少有東西）
        final_events = all_events

    return {"structured_events": final_events}


workflow = StateGraph(SupervisorState, input_schema=SupervisorStateInput)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("supervisor_tools", supervisor_tools_node)
workflow.add_node("structure_events", structure_events)
workflow.add_edge(START, "supervisor")

graph = workflow.compile().with_config({"callbacks": [get_langfuse_handler()]})
