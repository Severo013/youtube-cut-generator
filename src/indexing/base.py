"""Contract for transcript indexing providers."""

from typing import Protocol

from src.models import IndexResult, Transcript


class Indexer(Protocol):
    """Generate a structured episode index from a normalized transcript."""

    def index(self, transcript: Transcript, prompt: str) -> IndexResult:
        """Index ``transcript`` according to the fully rendered ``prompt``."""
        ...

