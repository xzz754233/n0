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
    structure_events_prompt,
)

# 引入子圖
from src.research_events.research_events_graph import research_events_app
from src.state import (
    FactCheckReport,  # [UPDATED] 取代 Chronology
    EvidencePoint,  # [UPDATED] 取代 ChronologyEvent
    FinishResearchTool,
    ResearchEventsTool,
    SupervisorState,
    SupervisorStateInput,
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

    # [UPDATED] 使用 evidence_points 進行摘要
    current_evidence = state.get("evidence_points", [])
    events_summary_text = (
        f"Currently have {len(current_evidence)} evidence points collected."
    )

    if current_evidence:
        # 提供最近收集到的幾個證據主題
        topics = [e.topic for e in current_evidence[-5:]]
        events_summary_text += f" Recent findings: {', '.join(topics)}"

    system_message = SystemMessage(
        content=lead_researcher_prompt.format(
            person_to_research=state["person_to_research"],
            events_summary=events_summary_text,
            last_message=last_message,
        )
    )

    human_message = HumanMessage(content="Start the investigation process.")
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

    # 用於累加這一輪新發現的證據
    newly_found_evidence = []

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
            print(f"Investigating: {research_question}")

            try:
                # 調用子圖
                result = await research_events_app.ainvoke(
                    {
                        "research_question": research_question,
                        "target_urls": [],
                        "processed_urls": [],
                        "gathered_events": [],
                    }
                )

                # 子圖回傳的是 RawEvent 列表
                raw_events = result.get("gathered_events", [])

                # [Data Transformation] RawEvent -> EvidencePoint
                # 我們將 RawEvent 轉換為 EvidencePoint 格式存入 State
                # 這裡做初步轉換，細緻的 Stance/Title 生成留給最後一步的 structure_events
                for raw in raw_events:
                    newly_found_evidence.append(
                        EvidencePoint(
                            id=str(uuid.uuid4())[:8],
                            # 暫時使用 Category 作為標題，讓最後一步 LLM 重寫
                            topic=f"Finding ({raw.category})",
                            details=raw.description,
                            # 暫時標記為 Pending，讓最後一步 LLM 判斷
                            stance="Pending Analysis",
                            source_title="Source",
                            source_url=raw.source_url,
                        )
                    )

                content_msg = f"Found {len(raw_events)} evidence points related to {research_question}."

            except Exception as e:
                print(f"Error in ResearchEventsTool: {e}")
                content_msg = f"Error executing research: {str(e)}"

            all_tool_messages.append(
                ToolMessage(content=content_msg, tool_call_id=tool_id, name=tool_name)
            )

    # 更新 State：將新發現的事件加入 evidence_points
    existing = state.get("evidence_points", [])
    if existing is None:
        existing = []

    updated_evidence = existing + newly_found_evidence

    return Command(
        goto="supervisor",
        update={
            "conversation_history": all_tool_messages,
            "evidence_points": updated_evidence,  # [UPDATED] Key Update
        },
    )


async def structure_events(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """
    Step 3: Final Consolidation (The Reduce Step).
    Converts raw findings into EvidencePoint objects with Citations and Verdicts.
    """
    print("--- Final Step: Weighing the Evidence & Generating Verdict ---")

    # [UPDATED] 使用正確的 Key
    all_raw_events = state.get("evidence_points", [])

    if not all_raw_events:
        return {"evidence_points": []}

    # 1. 準備輸入資料：將大量事件轉為文字，並 **附帶 URL**
    events_text_blob = ""
    for idx, e in enumerate(all_raw_events):
        # [CRITICAL] 這裡明確標註 Source URL，讓 Prompt 能抓到
        source_info = (
            f"(Source: {e.source_url})" if e.source_url else "(Source: Unknown)"
        )
        events_text_blob += (
            f"Finding {idx + 1} [{e.topic}]: {e.details} {source_info}\n"
        )

    # 2. 調用 LLM 生成最終 JSON
    # 使用 FactCheckReport 結構
    structured_llm = create_llm_structured_model(
        config=config, class_name=FactCheckReport
    )

    try:
        # 使用 structure_events_prompt (已在 prompts.py 更新為針對 Fact Check 的版本)
        final_result = await structured_llm.ainvoke(
            structure_events_prompt.format(existing_events=events_text_blob)
        )
        final_evidence = final_result.evidence_points

        # 簡單後處理 ID
        for e in final_evidence:
            if not e.id:
                e.id = str(uuid.uuid4())[:8]

        print(
            f"Final Verdict Dossier generated with {len(final_evidence)} evidence points."
        )

    except Exception as e:
        print(f"Error in final verdict generation: {e}")
        final_evidence = all_raw_events  # Fallback

    return {"evidence_points": final_evidence}


workflow = StateGraph(SupervisorState, input_schema=SupervisorStateInput)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("supervisor_tools", supervisor_tools_node)
workflow.add_node("structure_events", structure_events)
workflow.add_edge(START, "supervisor")

graph = workflow.compile().with_config({"callbacks": [get_langfuse_handler()]})
