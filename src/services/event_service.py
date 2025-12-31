from typing import List
from src.state import CategoriesWithEvents


class EventService:
    @staticmethod
    def split_events_into_chunks(
        extracted_events: str, max_len: int = 2000
    ) -> List[str]:
        """Split events text into chunks of specified length."""
        return [
            extracted_events[i : i + max_len]
            for i in range(0, len(extracted_events), max_len)
        ]

    @staticmethod
    def merge_categorized_events(
        categorized_results: List[CategoriesWithEvents],
    ) -> CategoriesWithEvents:
        """Merge multiple categorized event results into one."""
        # CHANGED: Updated fields to match Drama/Gossip domain
        # Also changed default from "[]" to "" to ensure clean bullet point concatenation
        merged = CategoriesWithEvents(
            context="",
            conflict="",
            reaction="",
            outcome="",
        )

        for result in categorized_results:
            merged.context += result.context
            merged.conflict += result.conflict
            merged.reaction += result.reaction
            merged.outcome += result.outcome

        return merged
