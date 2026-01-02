# src/research_events/research_events_graph.py
import asyncio
from typing import Literal

from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from src.configuration import Configuration
from src.services.event_service import EventService
from src.state import ResearchState
from src.url_crawler.utils import url_crawl, chunk_text_by_tokens
from src.utils import get_langfuse_handler
from src.services.url_service import URLService


# 1. 搜尋節點
def search_node(state: ResearchState) -> Command[Literal["process_batch"]]:
    """Find relevant URLs using Tavily."""
    question = state.get("research_question")
    existing_urls = state.get("processed_urls", [])

    # 使用 Tavily 搜尋
    tavily = TavilySearch(
        max_results=3, include_answer=False, include_raw_content=False
    )
    search_results = tavily.invoke({"query": question})

    found_urls = [r["url"] for r in search_results.get("results", [])]

    # 簡單過濾掉已經爬過的
    new_urls = [url for url in found_urls if url not in existing_urls]

    print(f"🔍 Found {len(new_urls)} new URLs for: {question}")

    return Command(goto="process_batch", update={"target_urls": new_urls})


# 2. 批次處理節點 (核心優化)
async def process_batch_node(
    state: ResearchState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """
    Crawl and Extract events from ALL target URLs in parallel.
    This replaces the old loop-based merge logic.
    """
    urls = state.get("target_urls", [])
    question = state.get("research_question", "")

    if not urls:
        return Command(goto=END)

    print(f"🚀 Batch processing {len(urls)} URLs...")

    # 定義單個 URL 的處理邏輯
    async def process_single_url(url):
        try:
            # A. 爬取
            content = await url_crawl(url)
            if not content:
                return []

            # B. 分塊
            chunks = await chunk_text_by_tokens(
                content, chunk_size=3000, overlap_size=100
            )

            # C. 提取 (這一步會調用 EventService 做並發提取)
            # 限制每個網頁最多看前 3-4 個 chunks，避免 token 浪費在 footer/側邊欄
            limit_chunks = chunks[:4]
            events = await EventService.run_batch_extraction(
                limit_chunks, url, question, config
            )
            return events
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")
            return []

    # 並發執行所有 URL 的處理
    results = await asyncio.gather(*[process_single_url(url) for url in urls])

    # 展平結果
    all_new_events = [e for batch in results for e in batch]

    print(f"📦 Batch complete. Total raw events extracted: {len(all_new_events)}")

    # 這裡利用 State 的 operator.add 自動將 all_new_events 加入 gathered_events
    return Command(
        goto=END,
        update={
            "gathered_events": all_new_events,
            "processed_urls": urls,  # 記錄已處理
            "target_urls": [],  # 清空待處理隊列
        },
    )


# --- Graph Definition ---
workflow = StateGraph(ResearchState)
workflow.add_node("search_node", search_node)
workflow.add_node("process_batch", process_batch_node)

workflow.add_edge(START, "search_node")
# search_node 直接指派了 goto="process_batch"，這裡不需要 edge，但為了視覺化可以加
# workflow.add_edge("search_node", "process_batch")
workflow.add_edge("process_batch", END)

research_events_app = workflow.compile().with_config(
    {"callbacks": [get_langfuse_handler()]}
)
