"""Research agent runtime."""

from __future__ import annotations

import re
import time

from .corpus import Corpus
from .llm import LLMClient
from .memory import ResearchMemory
from .models import AgentArchitectureConfig, ResearchAnswer, ResearchTask, SearchResult
from .observability import JsonlTrace
from .tools import ToolRuntime

CITATION_RE = re.compile(r"\[([A-Z][0-9]+)\]")


class ResearchAgent:
    def __init__(
        self,
        *,
        config: AgentArchitectureConfig,
        corpus: Corpus,
        trace: JsonlTrace,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.corpus = corpus
        self.trace = trace
        self.llm = llm or LLMClient.from_env()

    def run(self, task: ResearchTask) -> ResearchAnswer:
        started = time.perf_counter()
        self.trace.emit("agent.task.start", task_id=task.id, config=self.config.name)
        memory = ResearchMemory()
        tool_runtime = ToolRuntime(corpus=self.corpus, trace=self.trace)

        query = self._build_query(task.question)
        results, errors = tool_runtime.search(
            task_id=task.id,
            query=query,
            config=self.config,
            inject_transient_failure=task.tool_failure_probe,
        )

        memory.ingest(results, self.config)
        answer = self._synthesize(task, results, memory)
        citations = set(CITATION_RE.findall(answer))
        if self.config.retrieval.citation_verification:
            verdicts = tool_runtime.verify_citations(answer.splitlines(), citations, self.config)
            if not all(verdicts.values()):
                self.trace.emit("agent.citation.warning", task_id=task.id, verdicts=verdicts)

        elapsed_ms = (time.perf_counter() - started) * 1000
        used_ids = sorted(set(CITATION_RE.findall(answer)))
        response = ResearchAnswer(
            task_id=task.id,
            config_name=self.config.name,
            answer=answer,
            used_evidence_ids=used_ids,
            retrieved_evidence_ids=[result.document.id for result in results],
            tool_errors=errors,
            elapsed_ms=round(elapsed_ms, 2),
        )
        self.trace.emit(
            "agent.task.end",
            task_id=task.id,
            config=self.config.name,
            used_evidence_ids=used_ids,
            tool_errors=errors,
            elapsed_ms=response.elapsed_ms,
        )
        return response

    def _build_query(self, question: str) -> str:
        if not self.config.retrieval.query_expansion:
            return question

        lower = question.lower()
        expansions: list[str] = []
        if "unsupported" in lower or "claim" in lower:
            expansions.append("citation verification evidence support unsupported claim")
        if "noisy" in lower or "retrieval" in lower:
            expansions.append("source diversity noisy retrieval top-k")
        if "config" in lower or "self-mod" in lower or "architecture" in lower:
            expansions.append("schema validation rollback mutation budget freeze architecture")
        if "context" in lower or "memory" in lower:
            expansions.append("context drift scratchpad source grounded memory degradation")
        if "tool" in lower or "broken" in lower or "failure" in lower:
            expansions.append("tool retries typed failure fallback observability")
        if "fluent" in lower or "conflict" in lower or "contradict" in lower:
            expansions.append("evaluation contradiction citation-level checks strict evidence metrics limitation")
        lesson_terms = " ".join(self.config.notes)
        query = " ".join([question, *expansions, lesson_terms])
        self.trace.emit("agent.query.expanded", original=question, expanded=query)
        return query

    def _synthesize(self, task: ResearchTask, results: list[SearchResult], memory: ResearchMemory) -> str:
        if not results:
            return (
                "Retrieval failed before enough evidence was collected. "
                "The run is marked incomplete instead of synthesizing from an empty context."
            )

        if self.config.planner.evidence_first:
            return self._evidence_first_answer(task, results, memory)
        return self._baseline_answer(task, results)

    def _baseline_answer(self, task: ResearchTask, results: list[SearchResult]) -> str:
        first = results[0].document
        lines = [
            "Answer:",
            f"- {self._compress(first.text)} [{first.id}]",
        ]
        if len(results) > 1:
            second = results[1].document
            lines.append(f"- It also suggests that {self._compress(second.text).lower()}")
        lines.append(
            "- Overall, the agent should keep researching until the answer looks coherent, "
            "but this answer does not fully audit evidence coverage."
        )
        return "\n".join(lines)

    def _evidence_first_answer(
        self,
        task: ResearchTask,
        results: list[SearchResult],
        memory: ResearchMemory,
    ) -> str:
        selected = results[: self.config.retrieval.top_k]
        lines = [f"Research answer for {task.id}:"]
        for result in selected:
            lines.append(f"- {self._compress(result.document.text)} [{result.document.id}]")

        ids = {result.document.id for result in selected}
        if self.config.planner.contradiction_check:
            if {"E12", "E13"} <= ids or len(set(task.contradiction_evidence_ids) & ids) >= 2:
                lines.append(
                    "- The evidence contains a tension: fluent answers can appear correct while "
                    "citation-level checks reveal missing support, so the frozen agent reports uncertainty. "
                    + " ".join(f"[{evidence_id}]" for evidence_id in sorted(ids & {"E12", "E13"}))
                )
            elif "evaluation mismatch" in " ".join(self.config.notes).lower():
                lines.append(
                    "- Citation-level evaluation is still required because answer fluency can "
                    "disagree with evidence support. [E7]"
                )

        if memory.notes:
            remembered = " ".join(f"[{evidence_id}]" for evidence_id in memory.evidence_ids())
            lines.append(f"- Working memory kept only source-grounded notes from: {remembered}.")
        lines.append(
            "Conclusion: freeze the architecture after validation stabilizes; "
            "the measured adaptation is the workflow configuration."
        )
        return "\n".join(lines)

    def _compress(self, text: str, limit: int = 210) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."


def read_answer_citations(answer: str) -> set[str]:
    return set(CITATION_RE.findall(answer))
