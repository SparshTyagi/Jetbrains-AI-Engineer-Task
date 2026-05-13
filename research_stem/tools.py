"""Tool wrappers with typed failures and retry-visible behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Corpus
from .models import AgentArchitectureConfig, SearchResult
from .observability import JsonlTrace


class ToolCallError(RuntimeError):
    pass


@dataclass
class ToolRuntime:
    corpus: Corpus
    trace: JsonlTrace
    failure_injected: set[str] = field(default_factory=set)

    def search(
        self,
        *,
        task_id: str,
        query: str,
        config: AgentArchitectureConfig,
        inject_transient_failure: bool = False,
    ) -> tuple[list[SearchResult], list[str]]:
        tool = next((item for item in config.tools if item.name == "corpus_search" and item.enabled), None)
        if tool is None:
            raise ToolCallError("corpus_search tool is disabled")

        errors: list[str] = []
        attempts = tool.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                if inject_transient_failure and task_id not in self.failure_injected:
                    self.failure_injected.add(task_id)
                    raise ToolCallError("simulated transient corpus index timeout")

                results = self.corpus.search(
                    query,
                    top_k=config.retrieval.top_k,
                    source_diversity=config.retrieval.source_diversity,
                    max_per_source=config.retrieval.max_per_source,
                    min_score=config.retrieval.min_score,
                )
                self.trace.emit(
                    "tool.search.success",
                    task_id=task_id,
                    attempt=attempt,
                    top_k=config.retrieval.top_k,
                    result_ids=[result.document.id for result in results],
                )
                return results, errors
            except ToolCallError as exc:
                errors.append(str(exc))
                self.trace.emit(
                    "tool.search.failure",
                    task_id=task_id,
                    attempt=attempt,
                    error=str(exc),
                    retries_allowed=tool.retries,
                )
        return [], errors

    def verify_citations(
        self,
        answer_lines: list[str],
        evidence_ids: set[str],
        config: AgentArchitectureConfig,
    ) -> dict[str, bool]:
        enabled = any(tool.name == "citation_verifier" and tool.enabled for tool in config.tools)
        if not enabled:
            return {evidence_id: False for evidence_id in evidence_ids}

        verdicts = {evidence_id: evidence_id in self.corpus.by_id for evidence_id in evidence_ids}
        self.trace.emit(
            "tool.citation_verifier",
            evidence_ids=sorted(evidence_ids),
            verified=[key for key, value in verdicts.items() if value],
            answer_line_count=len(answer_lines),
        )
        return verdicts
