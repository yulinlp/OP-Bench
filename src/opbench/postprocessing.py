"""Optional memory-context post-processing methods used in the paper."""

from __future__ import annotations

from typing import Any

from .clients import chat_completion
from .config import EndpointConfig

FILTER_PROMPTS = {
    "self_recheck": """You filter retrieved memories for a conversational answer.
Keep only the original sentences that are directly useful for the question.
Do not answer the question, rewrite the memories, or invent information.
If nothing is useful, return exactly: NO RELEVANT CONTEXT""",
    "reminder": """Identify the parts of the supplied context that are directly useful for
answering the question. Return only the useful context; do not answer the question
and do not add information.""",
    # The original experiment called this method few-shot chain-of-thought. The
    # public implementation intentionally stores only the filtered evidence and
    # does not persist hidden reasoning traces.
    "few_shot_chain_of_thought": """Select the minimal set of context lines that can help answer
the question. Preserve the wording of selected lines. Return only those lines,
or NO RELEVANT CONTEXT when none apply. Do not answer the question.""",
    "comorag": """Act as a retrieval compressor. Keep the smallest set of independent
evidence statements that are relevant to the question. Preserve facts and wording;
do not answer the question.""",
    "marag": """Act as a memory-aware retrieval filter. Rank the context mentally, then
return only evidence that is both relevant to the question and sufficient for a
grounded answer. Preserve the original wording and do not answer the question.""",
}

METHODS = ("none", *FILTER_PROMPTS.keys(), "self_critic")


class ContextProcessor:
    def __init__(self, client: Any, endpoint: EndpointConfig):
        self.client = client
        self.endpoint = endpoint

    def filter(self, method: str, question: str, context: str) -> tuple[str, float, str | None]:
        if method in ("", "none") or not context.strip():
            return context, 0.0, None
        if method not in FILTER_PROMPTS:
            raise ValueError(f"Unknown context method: {method}")
        result = chat_completion(
            self.client,
            self.endpoint,
            [
                {"role": "system", "content": FILTER_PROMPTS[method]},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
            ],
            temperature=0.0,
            max_tokens=self.endpoint.max_tokens,
        )
        return result.text.strip(), result.duration_ms, result.error

    def revise(
        self, question: str, context: str, previous_answer: str
    ) -> tuple[str, float, str | None]:
        prompt = f"""Review the previous answer against the question and supplied memory.
Return a helpful revised answer. Use memory only when it is relevant, avoid
unnecessary personalization, and do not mention this review process.

Question: {question}
Memory:
{context}
Previous answer:
{previous_answer}
"""
        result = chat_completion(
            self.client,
            self.endpoint,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.endpoint.max_tokens,
        )
        return result.text.strip(), result.duration_ms, result.error
