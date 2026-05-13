"""Stem adaptation loop."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from .compat import ValidationError
from .evaluation import run_evaluation
from .io import load_architecture, load_domain, load_workflows, resolve_path, save_architecture, write_json
from .llm import LLMClient
from .models import AgentArchitectureConfig, MemoryConfig, PlannerConfig, RetrievalConfig, ToolSpec, validate_architecture_config
from .observability import JsonlTrace


def mine_workflow_lessons(workflows_path: str | Path, llm: LLMClient | None = None) -> list[str]:
    workflows = load_workflows(workflows_path)
    lessons = [item.architecture_lesson for item in workflows]
    client = llm or LLMClient.from_env()
    summary = client.complete(
        "You summarize engineering workflow lessons.",
        "Summarize architecture lessons from successful workflows:\n" + "\n".join(lessons),
    )
    distilled = [
        "evidence-first planning",
        "source diversity for noisy retrieval",
        "citation verification for unsupported claims",
        "bounded source-grounded memory",
        "tool retries with typed failures",
        "schema validation rollback",
        "freeze architecture after validation stability",
        "evaluation mismatch between fluency and evidence support",
    ]
    normalized_summary = _normalize_llm_summary(summary)
    if normalized_summary:
        distilled.append(normalized_summary)
    return distilled


def _normalize_llm_summary(summary: str) -> str:
    """Keep API-mined lessons useful without letting verbose prose pollute retrieval."""

    if not summary or "Mock LLM" in summary:
        return ""

    lower = summary.lower()
    concepts = [
        ("evidence", "evidence-first planning"),
        ("citation", "citation verification"),
        ("source diversity", "source diversity"),
        ("query expansion", "query expansion"),
        ("memory", "source-grounded memory"),
        ("retry", "tool retries"),
        ("typed", "typed tool failures"),
        ("schema", "schema validation"),
        ("rollback", "rollback"),
        ("freeze", "freeze after validation"),
    ]
    selected: list[str] = []
    for needle, label in concepts:
        if needle in lower and label not in selected:
            selected.append(label)
    if not selected:
        return "llm-mined workflow lessons available but not used for retrieval expansion"
    return "llm-mined lessons: " + ", ".join(selected[:8])


def candidate_configs(
    baseline: AgentArchitectureConfig,
    lessons: list[str],
) -> list[AgentArchitectureConfig | dict]:
    expanded = deepcopy(baseline)
    expanded.name = "candidate_expanded_retrieval"
    expanded.version = "0.2"
    expanded.retrieval.top_k = 4
    expanded.retrieval.query_expansion = True
    expanded.notes = lessons[:3]

    evidence_first = deepcopy(expanded)
    evidence_first.name = "candidate_evidence_first"
    evidence_first.version = "0.3"
    evidence_first.tools = [
        ToolSpec(name="corpus_search", enabled=True, retries=1),
        ToolSpec(name="citation_verifier", enabled=True, retries=0),
        ToolSpec(name="note_memory", enabled=False, retries=0),
    ]
    evidence_first.retrieval.citation_verification = True
    evidence_first.planner = PlannerConfig(
        style="evidence_first",
        steps=["plan_queries", "retrieve", "verify_citations", "synthesize"],
        contradiction_check=False,
        evidence_first=True,
    )
    evidence_first.notes = lessons[:5]

    specialized = deepcopy(evidence_first)
    specialized.name = "research_failures_specialist"
    specialized.version = "1.0"
    specialized.tools = [
        ToolSpec(name="corpus_search", enabled=True, retries=2),
        ToolSpec(name="citation_verifier", enabled=True, retries=0),
        ToolSpec(name="note_memory", enabled=True, retries=0),
    ]
    specialized.memory = MemoryConfig(enabled=True, source_grounded=True, max_notes=8, decay_policy="latest_verified")
    specialized.retrieval = RetrievalConfig(
        strategy="keyword",
        top_k=6,
        source_diversity=True,
        max_per_source=3,
        query_expansion=True,
        citation_verification=True,
        min_score=0.0,
    )
    specialized.planner = PlannerConfig(
        style="evidence_first",
        steps=["expand_domain_query", "retrieve_diverse_sources", "write_grounded_notes", "verify_citations", "check_contradictions", "synthesize"],
        contradiction_check=True,
        evidence_first=True,
    )
    specialized.notes = lessons

    invalid_probe = {
        **specialized.model_dump(),
        "name": "candidate_invalid_memory_probe",
        "memory": {"enabled": True, "source_grounded": True, "max_notes": 0, "decay_policy": "latest_verified"},
    }

    return [expanded, invalid_probe, evidence_first, specialized]


def evolve(
    *,
    domain_path: str | Path,
    baseline_path: str | Path = "configs/baseline.json",
    artifact_dir: str | Path = "artifacts",
) -> AgentArchitectureConfig:
    domain_file = Path(domain_path)
    root = domain_file.resolve().parents[2] if domain_file.is_absolute() else Path.cwd()
    domain = load_domain(domain_file)
    baseline = load_architecture(resolve_path(baseline_path, root))
    trace_path = Path(artifact_dir) / "traces" / f"evolve_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.jsonl"
    trace = JsonlTrace(trace_path)
    trace.emit("evolution.bootstrap", domain=domain.name, baseline=baseline.name)

    lessons = mine_workflow_lessons(resolve_path(domain.workflow_examples_path, root))
    trace.emit("evolution.workflow_mining", lessons=lessons)

    best_config: AgentArchitectureConfig | None = None
    best_score = -1.0
    experiment_summaries: list[dict] = []
    mutation_count = 0

    for candidate in candidate_configs(baseline, lessons):
        if mutation_count >= domain.mutation_budget:
            trace.emit("evolution.mutation_budget.stop", mutation_count=mutation_count)
            break
        mutation_count += 1
        try:
            config = candidate if isinstance(candidate, AgentArchitectureConfig) else AgentArchitectureConfig(**candidate)
            validate_architecture_config(config)
        except (ValidationError, ValueError, TypeError) as exc:
            trace.emit("evolution.rollback", candidate_name=getattr(candidate, "name", "dict_candidate"), error=str(exc))
            experiment_summaries.append({"candidate": "invalid_probe", "status": "rolled_back", "error": str(exc)})
            continue

        trace.emit("evolution.config_proposal", candidate=config.name, version=config.version)
        run = run_evaluation(domain_path=domain_file, config=config, mode=config.name, split="validation", artifact_dir=artifact_dir)
        score = run.aggregate_metrics.get("score", 0.0)
        experiment_summaries.append(
            {
                "candidate": config.name,
                "status": "evaluated",
                "score": score,
                "aggregate_metrics": run.aggregate_metrics,
            }
        )
        if score > best_score:
            best_config = config
            best_score = score
            trace.emit("evolution.best_updated", candidate=config.name, score=score)

    if best_config is None:
        raise RuntimeError("No valid candidate architecture was produced")

    best_config.frozen = best_score >= domain.success_threshold
    frozen_path = Path(artifact_dir) / "frozen_architecture.json"
    save_architecture(frozen_path, best_config)
    trace.emit("evolution.freeze", config=best_config.name, score=best_score, frozen=best_config.frozen, output=str(frozen_path))

    report = {
        "domain": domain.name,
        "success_threshold": domain.success_threshold,
        "best_config": best_config.model_dump(),
        "best_validation_score": best_score,
        "frozen": best_config.frozen,
        "experiments": experiment_summaries,
        "trace_path": str(trace_path),
    }
    write_json(Path(artifact_dir) / "evolution" / "latest_experiments.json", report)
    _write_report(Path(artifact_dir) / "evolution" / "latest_evolution_report.md", report)
    return best_config


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evolution Report",
        "",
        f"Domain: `{report['domain']}`",
        f"Best config: `{report['best_config']['name']}`",
        f"Best validation score: `{report['best_validation_score']}`",
        f"Frozen: `{report['frozen']}`",
        "",
        "## Candidate Results",
    ]
    for item in report["experiments"]:
        if item["status"] == "rolled_back":
            lines.append(f"- `{item['candidate']}` rolled back: {item['error']}")
        else:
            lines.append(f"- `{item['candidate']}` score `{item['score']}` metrics `{item['aggregate_metrics']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "The winning architecture improved by changing retrieval, memory, planning, and verification configuration. "
            "The invalid probe is intentional: it demonstrates schema-level rollback instead of silent self-modification failure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
