"""File loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from .compat import BaseModel
from .models import AgentArchitectureConfig, DomainSpec, ResearchTask, WorkflowExample, validate_architecture_config

T = TypeVar("T", bound=BaseModel)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base or project_root()) / candidate


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Read the intentionally small YAML subset used by this project."""

    data: dict[str, Any] = {}
    current_list_key: str | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- ") and current_list_key:
                data[current_list_key].append(_parse_scalar(stripped[2:]))
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                data[key] = []
                current_list_key = key
            else:
                data[key] = _parse_scalar(value)
                current_list_key = None
    return data


def load_domain(path: str | Path) -> DomainSpec:
    try:
        import yaml  # type: ignore

        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except ModuleNotFoundError:
        data = read_simple_yaml(path)
    return DomainSpec(**data)


def load_architecture(path: str | Path) -> AgentArchitectureConfig:
    config = AgentArchitectureConfig(**read_json(path))
    validate_architecture_config(config)
    return config


def save_architecture(path: str | Path, config: AgentArchitectureConfig) -> None:
    validate_architecture_config(config)
    write_json(path, config.model_dump())


def load_tasks(path: str | Path, split: str | None = None) -> list[ResearchTask]:
    tasks = [ResearchTask(**row) for row in read_jsonl(path)]
    if split:
        return [task for task in tasks if task.split == split]
    return tasks


def load_workflows(path: str | Path) -> list[WorkflowExample]:
    return [WorkflowExample(**row) for row in read_jsonl(path)]
