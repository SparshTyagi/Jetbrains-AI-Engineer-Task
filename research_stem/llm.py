"""OpenAI-compatible client with a deterministic mock default."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMClient:
    mode: str = "mock"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_env(cls) -> "LLMClient":
        mode = os.getenv("RESEARCH_STEM_LLM", "mock").strip().lower()
        return cls(
            mode=mode,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        )

    def complete(self, system: str, user: str) -> str:
        if self.mode != "api":
            return self._mock_complete(system, user)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when RESEARCH_STEM_LLM=api")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310 - user-provided API endpoint.
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def _mock_complete(self, system: str, user: str) -> str:
        if "architecture lessons" in user.lower():
            return (
                "Prefer evidence-first planning, source-diverse retrieval, citation verification, "
                "bounded source-grounded memory, tool retries, schema validation, and freeze-on-validation."
            )
        return "Mock LLM mode: deterministic extractive synthesis is handled by the local agent runtime."
