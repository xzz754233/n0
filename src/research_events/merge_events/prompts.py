categorize_events_prompt = """
You are a helpful assistant that will categorize the events into the 4 categories.

<Events>
{events}
</Events>

<Categories>
context: Background info, previous relationships, or the 'calm before the storm'. origin of the beef.
conflict: The main incident, the accusation, the leak, the breakup, or the scandal itself.
reaction: Public responses, PR statements, tweets from other influencers, lawsuits, or 'receipts' posted.
outcome: Current status, who was cancelled, impact on career, or final resolution (if any).
</Categories>


<Rules>
INCLUDE ALL THE INFORMATION FROM THE EVENTS, do not abbreviate or omit any information.
</Rules>
"""

EXTRACT_AND_CATEGORIZE_PROMPT = """
You are a Drama/Scandal Event Extractor and Categorizer. Your task is to analyze text chunks for events related to the topic/person.**

<Available Tools>
- `IrrelevantChunk` (use if the text contains NO drama/scandal events relevant to the research question)
- `RelevantEventsCategorized` (use if the text contains relevant events - categorize them into the 4 categories)
</Available Tools>

<Categories>
context: Background info, previous relationships, or the 'calm before the storm'. origin of the beef.
conflict: The main incident, the accusation, the leak, the breakup, or the scandal itself.
reaction: Public responses, PR statements, tweets from other influencers, lawsuits, or 'receipts' posted.
outcome: Current status, who was cancelled, impact on career, or final resolution (if any).
</Categories>

**EXTRACTION RULES**:
- Extract COMPLETE sentences with ALL available details (dates, names, platforms, context, emotions)
- Include "receipts" (screenshots, quotes, specific evidence)
- Preserve the tone of the drama (e.g., if it was a heated argument, describe it as such)
- Include only events directly relevant to the research question
- Maintain chronological order within each category
- Format as clean bullet points with complete, detailed descriptions
- IMPORTANT: Return each category as a SINGLE string containing all bullet points, not as a list

<Text to Analyze>
{text_chunk}
</Text to Analyze>

You must call exactly one of the provided tools. Do not respond with plain text.
"""


MERGE_EVENTS_TEMPLATE = """You are a helpful assistant that will merge two lists of events: 
the original events (which must always remain) and new events (which may contain extra details). 
The new events should only be treated as additions if they provide relevant new information. 
The final output must preserve the original events and seamlessly add the new ones if applicable.

<Rules>
- Always include the original events exactly, do not omit or alter them.
- Add new events only if they are not duplicates, combining details if they overlap.
- Format the final list as bullet points, one event per line (e.g., "- Event details.").
- Keep the list clean, concise, and without commentary.
</Rules>

<Events>
Original events:
{original}

New events:
{new}
</Events>

<Output>
Return only the merged list of events as bullet points, nothing else.
</Output>"""
