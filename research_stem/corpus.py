"""Local corpus search."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from .io import read_jsonl
from .models import CorpusDocument, SearchResult

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")
STOPWORDS = {
    "about",
    "after",
    "agent",
    "agents",
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "its",
    "should",
    "that",
    "the",
    "their",
    "this",
    "when",
    "where",
    "while",
    "with",
    "why",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


class Corpus:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.documents = [CorpusDocument(**row) for row in read_jsonl(self.path)]
        self.by_id = {document.id: document for document in self.documents}

    def search(
        self,
        query: str,
        *,
        top_k: int,
        source_diversity: bool = False,
        max_per_source: int = 2,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        query_terms = tokenize(query)
        query_counts = Counter(query_terms)
        results: list[SearchResult] = []

        for document in self.documents:
            document_terms = tokenize(" ".join([document.title, document.text, " ".join(document.tags)]))
            doc_counts = Counter(document_terms)
            matched_terms = sorted(set(query_counts) & set(doc_counts))
            if not matched_terms:
                continue
            overlap = sum(min(query_counts[term], doc_counts[term]) for term in matched_terms)
            phrase_bonus = 0.0
            lower_text = document.text.lower()
            for term in matched_terms:
                if term in lower_text:
                    phrase_bonus += 0.2
            tag_bonus = len(set(document.tags) & set(query_terms)) * 0.5
            score = overlap + phrase_bonus + tag_bonus
            if score >= min_score:
                results.append(SearchResult(document=document, score=round(score, 4), matched_terms=matched_terms))

        results.sort(key=lambda item: (-item.score, item.document.id))
        if not source_diversity:
            return results[:top_k]

        diversified: list[SearchResult] = []
        per_source: dict[str, int] = defaultdict(int)
        for result in results:
            if per_source[result.document.source] >= max_per_source:
                continue
            diversified.append(result)
            per_source[result.document.source] += 1
            if len(diversified) >= top_k:
                break
        return diversified
