"""Centralized prompt templates for the EDEL pipeline."""

from __future__ import annotations

DEFAULT_ASPECT_DEFINITIONS = {
    "problem": "The main research objective, problem, or question investigated in the study.",
    "method": "The strategy, method, model, inference reasoning, or analytical approach used.",
    "finding": "The main results, discoveries, or outcomes produced by the study.",
    "interpretation": "The conclusions, implications, or explanations offered for the findings.",
}

STRUCTURED_ABSTRACT_PROMPT_TEMPLATE = """
You are an expert research assistant. {topic_instruction}

You will receive the title, keywords, and abstract of a scientific paper.

Your task is to extract text from the abstract corresponding to four epistemic aspects.

Definitions:

1. Problem / Research Question
{problem_def}

2. Methods / Evidence
{method_def}

3. Findings / Results
{finding_def}

4. Interpretation / Discussion
{interpretation_def}

Rules:

- Only extract text that appears in the original abstract.
- You may extract full sentences or sentence fragments.
- Do NOT paraphrase or invent new text.
- The same text may appear in multiple categories if relevant.
- Prefer partial evidence over returning UNKNOWN.
- Return "UNKNOWN" only if the abstract contains no evidence for that aspect.

Return your answer as valid JSON:

{{
  "problem": "...",
  "method": "...",
  "finding": "...",
  "interpretation": "..."
}}

Title:
{title}

Abstract:
{abstract_text}

Keywords:
{keywords_str}

JSON Answer:
"""


def create_structuring_prompt(
    title: str,
    abstract_text: str,
    keywords: list[str] | str,
    topic: str | None = None,
    definitions: dict[str, str] | None = None,
) -> str:
    """Create a prompt for abstract segmentation into epistemic aspects."""
    topic_instruction = f"Assume the topic is {topic}." if topic else ""
    keywords_str = (
        ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
    )
    defs = definitions or DEFAULT_ASPECT_DEFINITIONS

    return STRUCTURED_ABSTRACT_PROMPT_TEMPLATE.format(
        topic_instruction=topic_instruction,
        problem_def=defs["problem"],
        method_def=defs["method"],
        finding_def=defs["finding"],
        interpretation_def=defs["interpretation"],
        title=title,
        abstract_text=abstract_text,
        keywords_str=keywords_str,
    )
