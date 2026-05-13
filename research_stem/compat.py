"""Small compatibility layer around Pydantic.

The submission prefers Pydantic models, and `pyproject.toml` declares Pydantic
as a dependency. The local review environment used while building this project
may not have third-party packages installed, so this module supplies a minimal
fallback with the subset of the API used by the project. When Pydantic is
installed, the real implementation is used.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from typing import Any, get_args, get_origin, get_type_hints


try:  # pragma: no cover - exercised when the dependency is installed.
    from pydantic import BaseModel, Field, ValidationError

    PYDANTIC_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - local dependency-free path.
    PYDANTIC_AVAILABLE = False

    class ValidationError(ValueError):
        """Fallback validation error."""

    class _FieldSpec:
        def __init__(
            self,
            default: Any = None,
            *,
            default_factory: Any = None,
            description: str | None = None,
            ge: float | None = None,
            le: float | None = None,
        ) -> None:
            self.default = default
            self.default_factory = default_factory
            self.description = description
            self.ge = ge
            self.le = le

        def value(self) -> Any:
            if self.default_factory is not None:
                return self.default_factory()
            return deepcopy(self.default)

    def Field(
        default: Any = None,
        *,
        default_factory: Any = None,
        description: str | None = None,
        ge: float | None = None,
        le: float | None = None,
    ) -> Any:
        return _FieldSpec(
            default,
            default_factory=default_factory,
            description=description,
            ge=ge,
            le=le,
        )

    def _all_annotations(cls: type) -> dict[str, Any]:
        annotations: dict[str, Any] = {}
        for base in reversed(cls.__mro__):
            module = sys.modules.get(base.__module__)
            globalns = vars(module) if module else None
            try:
                annotations.update(get_type_hints(base, globalns=globalns))
            except Exception:
                annotations.update(getattr(base, "__annotations__", {}))
        return annotations

    def _is_base_model_type(annotation: Any) -> bool:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)

    def _coerce(annotation: Any, value: Any) -> Any:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if value is None:
            return None
        if origin in (list, tuple):
            inner = args[0] if args else Any
            return [_coerce(inner, item) for item in value]
        if origin is dict:
            value_type = args[1] if len(args) > 1 else Any
            return {key: _coerce(value_type, item) for key, item in value.items()}
        if origin is not None and str(origin).endswith("Union"):
            for option in args:
                if option is type(None):
                    continue
                try:
                    return _coerce(option, value)
                except Exception:
                    continue
            return value
        if _is_base_model_type(annotation) and isinstance(value, dict):
            return annotation(**value)
        return value

    def _dump(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [_dump(item) for item in value]
        if isinstance(value, dict):
            return {key: _dump(item) for key, item in value.items()}
        return value

    class BaseModel:
        """Very small subset of Pydantic's BaseModel used by this project."""

        def __init__(self, **data: Any) -> None:
            annotations = _all_annotations(type(self))
            for name, annotation in annotations.items():
                class_default = getattr(type(self), name, None)
                if name in data:
                    value = data[name]
                elif isinstance(class_default, _FieldSpec):
                    value = class_default.value()
                elif class_default is not None:
                    value = deepcopy(class_default)
                else:
                    raise ValidationError(f"Missing required field: {name}")

                if isinstance(class_default, _FieldSpec):
                    if class_default.ge is not None and value < class_default.ge:
                        raise ValidationError(f"{name} must be >= {class_default.ge}")
                    if class_default.le is not None and value > class_default.le:
                        raise ValidationError(f"{name} must be <= {class_default.le}")
                setattr(self, name, _coerce(annotation, value))

            extra = set(data) - set(annotations)
            for name in extra:
                setattr(self, name, data[name])

            validator = getattr(self, "validate", None)
            if callable(validator):
                validator()

        @classmethod
        def model_validate(cls, data: dict[str, Any]) -> "BaseModel":
            return cls(**data)

        @classmethod
        def model_validate_json(cls, data: str) -> "BaseModel":
            return cls(**json.loads(data))

        def model_dump(self, **_: Any) -> dict[str, Any]:
            annotations = _all_annotations(type(self))
            return {name: _dump(getattr(self, name)) for name in annotations}

        def model_dump_json(self, indent: int | None = None, **_: Any) -> str:
            return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)
