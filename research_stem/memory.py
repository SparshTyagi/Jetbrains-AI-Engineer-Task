"""Source-grounded note memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import AgentArchitectureConfig, SearchResult


@dataclass
class ResearchNote:
    evidence_id: str
    source: str
    claim: str


@dataclass
class ResearchMemory:
    notes: list[ResearchNote] = field(default_factory=list)

    def ingest(self, results: list[SearchResult], config: AgentArchitectureConfig) -> None:
        if not config.memory.enabled:
            return
        for result in results:
            sentence = result.document.text.split(".")[0].strip()
            if not sentence:
                sentence = result.document.text.strip()
            if config.memory.source_grounded and not result.document.id:
                continue
            self.notes.append(
                ResearchNote(
                    evidence_id=result.document.id,
                    source=result.document.source,
                    claim=sentence,
                )
            )
        if config.memory.max_notes:
            self.notes = self.notes[-config.memory.max_notes :]

    def evidence_ids(self) -> list[str]:
        return [note.evidence_id for note in self.notes]
