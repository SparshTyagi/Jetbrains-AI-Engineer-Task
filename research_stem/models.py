"""Typed contracts for the stem-agent runtime."""

from __future__ import annotations

from typing import Any

from .compat import BaseModel, Field


class ArchitectureValidationError(ValueError):
    """Raised when a syntactically valid config violates runtime assumptions."""


class DomainSpec(BaseModel):
    name: str
    description: str
    corpus_path: str
    workflow_examples_path: str
    eval_path: str
    success_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    mutation_budget: int = Field(default=4, ge=1)
    concepts: list[str] = Field(default_factory=list)


class ResearchTask(BaseModel):
    id: str
    split: str
    question: str
    expected_evidence_ids: list[str] = Field(default_factory=list)
    required_concepts: list[str] = Field(default_factory=list)
    contradiction_evidence_ids: list[str] = Field(default_factory=list)
    tool_failure_probe: bool = False


class ToolSpec(BaseModel):
    name: str
    enabled: bool = True
    retries: int = Field(default=0, ge=0)
    timeout_ms: int = Field(default=3000, ge=100)


class MemoryConfig(BaseModel):
    enabled: bool = False
    source_grounded: bool = False
    max_notes: int = Field(default=0, ge=0)
    decay_policy: str = "none"


class RetrievalConfig(BaseModel):
    strategy: str = "keyword"
    top_k: int = Field(default=2, ge=1)
    source_diversity: bool = False
    max_per_source: int = Field(default=2, ge=1)
    query_expansion: bool = False
    citation_verification: bool = False
    min_score: float = Field(default=0.0, ge=0.0)


class PlannerConfig(BaseModel):
    style: str = "direct"
    steps: list[str] = Field(default_factory=lambda: ["answer_directly"])
    contradiction_check: bool = False
    evidence_first: bool = False


class AgentArchitectureConfig(BaseModel):
    name: str
    version: str = "0.1"
    tools: list[ToolSpec] = Field(default_factory=list)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    eval_metrics: list[str] = Field(default_factory=list)
    frozen: bool = False
    notes: list[str] = Field(default_factory=list)


class CorpusDocument(BaseModel):
    id: str
    source: str
    title: str
    text: str
    tags: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    document: CorpusDocument
    score: float
    matched_terms: list[str] = Field(default_factory=list)


class WorkflowExample(BaseModel):
    id: str
    domain_signal: str
    successful_pattern: str
    failure_observed: str
    architecture_lesson: str


class ResearchAnswer(BaseModel):
    task_id: str
    config_name: str
    answer: str
    used_evidence_ids: list[str] = Field(default_factory=list)
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0


class EvalResult(BaseModel):
    task_id: str
    mode: str
    score: float
    metrics: dict[str, float] = Field(default_factory=dict)
    expected_evidence_ids: list[str] = Field(default_factory=list)
    found_evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence_ids: list[str] = Field(default_factory=list)
    unsupported_claims: int = 0
    answer: str = ""


class ExperimentRun(BaseModel):
    id: str
    mode: str
    config_name: str
    split: str
    results: list[EvalResult] = Field(default_factory=list)
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    started_at: str
    completed_at: str
    trace_path: str


def validate_architecture_config(config: AgentArchitectureConfig) -> None:
    """Apply semantic checks that are useful even with the fallback models."""

    enabled_tools = {tool.name for tool in config.tools if tool.enabled}
    if "corpus_search" not in enabled_tools:
        raise ArchitectureValidationError("architecture must enable corpus_search")
    if config.retrieval.top_k < 1:
        raise ArchitectureValidationError("retrieval.top_k must be at least 1")
    if config.memory.enabled and config.memory.max_notes < 1:
        raise ArchitectureValidationError("enabled memory requires max_notes >= 1")
    if config.retrieval.citation_verification and "citation_verifier" not in enabled_tools:
        raise ArchitectureValidationError("citation verification requires citation_verifier tool")
    if config.planner.contradiction_check and not config.planner.evidence_first:
        raise ArchitectureValidationError("contradiction checks require evidence-first planning")
    if not config.eval_metrics:
        raise ArchitectureValidationError("architecture must declare evaluation metrics")


def average_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: sum(item.get(key, 0.0) for item in items) / len(items) for key in keys}


def dump_model_json(model: BaseModel | dict[str, Any]) -> str:
    if isinstance(model, BaseModel):
        return model.model_dump_json(indent=2)
    import json

    return json.dumps(model, indent=2, ensure_ascii=False)
