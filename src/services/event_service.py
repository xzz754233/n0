# src/services/event_service.py
import asyncio
from typing import List
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.state import RawEvent
from src.llm_service import create_llm_structured_model
from src.prompts import EVENT_EXTRACTION_PROMPT


# 定義一個 Wrapper Class，幫助 LLM 穩定輸出 List
class RawEventList(BaseModel):
    events: List[RawEvent] = Field(default_factory=list)


class EventService:
    @staticmethod
    async def extract_events_from_chunk(
        chunk: str, source_url: str, topic: str, config: RunnableConfig
    ) -> List[RawEvent]:
        """
        核心提取函數：
        輸入：一段文字 chunk
        輸出：結構化的事件列表 (RawEvent)
        """
        # 1. 快速檢查：如果 chunk 太短或無意義，直接跳過 (節省 LLM 成本)
        if len(chunk.strip()) < 50:
            return []

        # 2. 使用 Gemeni Flash (或配置的模型) 進行結構化提取
        # 注意：這裡我們使用 RawEventList 作為 schema，強迫模型輸出列表
        llm = create_llm_structured_model(config, class_name=RawEventList)

        try:
            # 3. 執行調用
            result = await llm.ainvoke(
                EVENT_EXTRACTION_PROMPT.format(text_chunk=chunk, topic=topic)
            )

            # 4. 後處理：注入來源 URL (這步很重要，解決幻覺問題)
            enriched_events = []
            if result and result.events:
                for event in result.events:
                    # 強制寫入來源，方便後續查證
                    event.source_url = source_url
                    # 簡單的清理
                    if event.description:
                        event.description = event.description.strip()
                    enriched_events.append(event)

            return enriched_events

        except Exception as e:
            # 容錯處理：如果提取失敗，不要讓整個程式崩潰，只打印錯誤並返回空列表
            print(f"⚠️ Extraction error for chunk from {source_url}: {e}")
            return []

    @staticmethod
    async def run_batch_extraction(
        chunks: List[str], source_url: str, topic: str, config: RunnableConfig
    ) -> List[RawEvent]:
        """
        批次處理函數：同時處理多個 chunk，極大化速度
        """
        if not chunks:
            return []

        # 建立任務列表
        tasks = [
            EventService.extract_events_from_chunk(chunk, source_url, topic, config)
            for chunk in chunks
        ]

        # 並發執行 (asyncio.gather) - 這是速度提升的關鍵
        results = await asyncio.gather(*tasks)

        # 展平結果 (Flatten List of Lists)
        # [[e1, e2], [e3], []] -> [e1, e2, e3]
        all_events = [event for batch in results for event in batch]

        # Log 用於調試
        if all_events:
            print(f"✅ Extracted {len(all_events)} events from {source_url}")
        else:
            print(f"⚠️ No events found in {source_url}")

        return all_events
