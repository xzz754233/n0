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


# 1. 搜尋節點：三角驗證法
def search_node(state: ResearchState) -> Command[Literal["process_batch"]]:
    """
    Finds evidence to confirm or bust a myth.
    Strategy: Triangulate the truth using Scientific, Statistical, and Debunking queries.
    """
    claim = state.get("research_question")
    existing_urls = state.get("processed_urls", [])

    # === MythBuster Search Strategy ===

    # 1. The Scientific/Data Angle (找硬證據)
    # 關鍵字：meta-analysis (統合分析) 是證據等級最高的
    query_science = f"{claim} scientific study meta-analysis statistics data"

    # 2. The Debunking Angle (找闢謠)
    query_debunk = f"{claim} myth debunked fact check true or false"

    # 3. The Origin/Context Angle (找起源)
    query_context = f"Why do people believe {claim}? origin of theory"

    queries = [query_science, query_debunk, query_context]

    print(f"MythBuster investigating claim: {claim}")

    tavily = TavilySearch(
        max_results=3, include_answer=False, include_raw_content=False
    )

    all_found_urls = []

    for q in queries:
        try:
            results = tavily.invoke({"query": q})
            urls = [r["url"] for r in results.get("results", [])]
            all_found_urls.extend(urls)
        except Exception as e:
            print(f"Search error for query '{q}': {e}")

    unique_found_urls = list(set(all_found_urls))
    new_urls = [url for url in unique_found_urls if url not in existing_urls]

    print(f"Found {len(new_urls)} new sources for verification.")

    return Command(goto="process_batch", update={"target_urls": new_urls})


# 2. 批次處理節點
async def process_batch_node(
    state: ResearchState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """
    Crawl and Extract evidence from ALL target URLs in parallel.
    """
    urls = state.get("target_urls", [])
    claim = state.get("research_question", "")

    if not urls:
        return Command(goto=END)

    print(f"Batch processing {len(urls)} sources...")

    async def process_single_url(url):
        try:
            content = await url_crawl(url)
            if not content:
                return []

            # 3000 chars 確保有足夠上下文判斷研究結果
            chunks = await chunk_text_by_tokens(
                content, chunk_size=3000, overlap_size=100
            )

            limit_chunks = chunks[:4]
            events = await EventService.run_batch_extraction(
                limit_chunks, url, claim, config
            )
            return events
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return []

    results = await asyncio.gather(*[process_single_url(url) for url in urls])
    all_new_evidence = [e for batch in results for e in batch]

    print(f"Batch complete. Total evidence points extracted: {len(all_new_evidence)}")

    return Command(
        goto=END,
        update={
            "gathered_events": all_new_evidence,
            "processed_urls": urls,
            "target_urls": [],
        },
    )


workflow = StateGraph(ResearchState)
workflow.add_node("search_node", search_node)
workflow.add_node("process_batch", process_batch_node)
workflow.add_edge(START, "search_node")
workflow.add_edge("process_batch", END)

research_events_app = workflow.compile().with_config(
    {"callbacks": [get_langfuse_handler()]}
)
