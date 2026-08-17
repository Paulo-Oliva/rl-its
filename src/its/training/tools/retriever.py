"""
The Retriever, which looks up reference material about a concept before the tutor explains it.
"""

import logging

from its.training.tools.base import Tool, ToolContext, ToolResult

log = logging.getLogger(__name__)


class RetrieverTool(Tool):
    """
    Retrieves a reference snippet about a math concept from a knowledge base.
    """

    name = "call_retriever"
    description = "Look up reference material about a math concept before explaining it."
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "the concept to look up"
            }
        },
        "required": ["topic"],
    }

    def __init__(self, kb: dict[str, str]) -> None:
        # The explain_concept scenario has to be given this same kb
        self.kb = kb

    def _lookup(self, topic: str) -> tuple[str, str] | None:
        q = topic.lower().strip()
        if not q:
            return None
        # An exact key first
        if q in self.kb:
            return q, self.kb[q]
        # Then a substring match in either direction
        for k, v in self.kb.items():
            if q in k or k in q:
                return k, v
        # And finally the key sharing the most words with the query
        qwords = set(q.split())
        best, best_key = 0, None
        for k in self.kb:
            overlap = len(qwords & set(k.split()))
            if overlap > best:
                best, best_key = overlap, k

        return (best_key, self.kb[best_key]) if best_key is not None else None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        topic = str(args.get("topic", "")).strip()
        if not topic:
            return ToolResult('error: provide the concept in {"topic": "..."}', ok=False)

        hit = self._lookup(topic)
        if hit is None:
            return ToolResult(f"error: no reference material found for {topic!r}", ok=False)

        matched, snippet = hit

        return ToolResult(text=f"Reference on {matched}: {snippet}", data={"topic": matched, "snippet": snippet})
