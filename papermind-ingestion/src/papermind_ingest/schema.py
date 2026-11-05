"""Typed records for a scientific-claim retrieval benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field

LABELS = ("SUPPORT", "CONTRADICT")


@dataclass(frozen=True)
class Document:
    doc_id: int
    title: str
    sentences: tuple[str, ...]

    @property
    def abstract(self) -> str:
        return " ".join(self.sentences)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract}" if self.title else self.abstract


@dataclass(frozen=True)
class Evidence:
    label: str
    sentences: tuple[int, ...]


@dataclass(frozen=True)
class Claim:
    claim_id: int
    text: str
    evidence: dict[int, tuple[Evidence, ...]] = field(default_factory=dict)
    cited_doc_ids: tuple[int, ...] = ()

    @property
    def relevant_docs(self) -> frozenset[int]:
        return frozenset(self.evidence)

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)


@dataclass
class Benchmark:
    corpus: dict[int, Document]
    train: list[Claim]
    dev: list[Claim]
    test: list[Claim]  # the public SciFact test labels are withheld: evidence is empty
