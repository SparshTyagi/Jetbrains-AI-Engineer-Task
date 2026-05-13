"""Deterministic evaluation for before/after experiments."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from .agent import ResearchAgent, read_answer_citations
from .corpus import Corpus
from .io import load_architecture, load_domain, load_tasks, resolve_path, write_json
from .models import AgentArchitectureConfig, EvalResult, ExperimentRun, ResearchAnswer, ResearchTask
from .observability import JsonlTrace


def _claim_lines(answer: str) -> list[str]:
    return [line for line in answer.splitlines() if line.strip().startswith("-") or line.startswith("Answer:")]


def score_answer(task: ResearchTask, response: ResearchAnswer, mode: str) -> EvalResult:
    citations = read_answer_citations(response.answer)
    expected = set(task.expected_evidence_ids)
    found_expected = sorted(expected & citations)
    missing = sorted(expected - citations)
    citation_precision = len(citations & set(response.retrieved_evidence_ids)) / max(len(citations), 1)
    evidence_recall = len(found_expected) / max(len(expected), 1)

    claim_lines = _claim_lines(response.answer)
    unsupported_claims = sum(1 for line in claim_lines if not read_answer_citations(line))
    unsupported_claim_rate = unsupported_claims / max(len(claim_lines), 1)

    lower = response.answer.lower()
    concept_hits = sum(1 for concept in task.required_concepts if concept.lower() in lower)
    rubric_score = concept_hits / max(len(task.required_concepts), 1)

    if task.contradiction_evidence_ids:
        cited_contradictions = set(task.contradiction_evidence_ids) & citations
        contradiction_handling = 1.0 if len(cited_contradictions) >= 2 and any(
            word in lower for word in ["tension", "contradict", "disagree", "uncertainty"]
        ) else 0.0
    else:
        contradiction_handling = 1.0

    if task.tool_failure_probe:
        tool_failure_recovery = 1.0 if response.tool_errors and response.used_evidence_ids else 0.0
    else:
        tool_failure_recovery = 1.0

    metrics = {
        "citation_precision": round(citation_precision, 4),
        "evidence_recall": round(evidence_recall, 4),
        "unsupported_claim_rate": round(unsupported_claim_rate, 4),
        "contradiction_handling": round(contradiction_handling, 4),
        "rubric_score": round(rubric_score, 4),
        "tool_failure_recovery": round(tool_failure_recovery, 4),
        "latency_ms": round(response.elapsed_ms, 2),
    }
    score = (
        0.28 * evidence_recall
        + 0.22 * citation_precision
        + 0.18 * (1.0 - unsupported_claim_rate)
        + 0.14 * contradiction_handling
        + 0.12 * rubric_score
        + 0.06 * tool_failure_recovery
    )
    return EvalResult(
        task_id=task.id,
        mode=mode,
        score=round(score, 4),
        metrics=metrics,
        expected_evidence_ids=task.expected_evidence_ids,
        found_evidence_ids=found_expected,
        missing_evidence_ids=missing,
        unsupported_claims=unsupported_claims,
        answer=response.answer,
    )


def aggregate_results(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {}
    metric_keys = sorted({key for result in results for key in result.metrics})
    aggregate = {key: round(mean(result.metrics.get(key, 0.0) for result in results), 4) for key in metric_keys}
    aggregate["score"] = round(mean(result.score for result in results), 4)
    return aggregate


def run_evaluation(
    *,
    domain_path: str | Path,
    config: AgentArchitectureConfig,
    mode: str,
    split: str,
    artifact_dir: str | Path = "artifacts",
) -> ExperimentRun:
    domain = load_domain(domain_path)
    root = Path(domain_path).resolve().parents[2] if Path(domain_path).is_absolute() else Path.cwd()
    corpus = Corpus(resolve_path(domain.corpus_path, root))
    tasks = load_tasks(resolve_path(domain.eval_path, root), split=split)

    started_at = datetime.now(UTC).isoformat()
    run_id = f"{mode}_{split}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    trace_path = Path(artifact_dir) / "traces" / f"{run_id}.jsonl"
    trace = JsonlTrace(trace_path)
    trace.emit("evaluation.start", mode=mode, split=split, config=config.name)

    agent = ResearchAgent(config=config, corpus=corpus, trace=trace)
    results: list[EvalResult] = []
    for task in tasks:
        response = agent.run(task)
        result = score_answer(task, response, mode=mode)
        results.append(result)
        trace.emit("evaluation.task.scored", task_id=task.id, score=result.score, metrics=result.metrics)

    aggregate = aggregate_results(results)
    completed_at = datetime.now(UTC).isoformat()
    run = ExperimentRun(
        id=run_id,
        mode=mode,
        config_name=config.name,
        split=split,
        results=results,
        aggregate_metrics=aggregate,
        started_at=started_at,
        completed_at=completed_at,
        trace_path=str(trace_path),
    )
    output = Path(artifact_dir) / "evaluations" / f"{run_id}.json"
    latest = Path(artifact_dir) / "evaluations" / f"latest_{mode}_{split}.json"
    write_json(output, run.model_dump())
    write_json(latest, run.model_dump())
    trace.emit("evaluation.end", aggregate=aggregate, output=str(output))
    return run


def run_evaluation_from_files(
    *,
    domain_path: str | Path,
    config_path: str | Path,
    mode: str,
    split: str,
    artifact_dir: str | Path = "artifacts",
) -> ExperimentRun:
    return run_evaluation(
        domain_path=domain_path,
        config=load_architecture(config_path),
        mode=mode,
        split=split,
        artifact_dir=artifact_dir,
    )
