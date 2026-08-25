"""Contract for transcript indexing providers."""

from typing import Protocol

from src.models import IndexResult, Transcript


class IndexingError(Exception):
    """Base error raised by indexing providers."""


class IndexingConfigurationError(IndexingError):
    """Raised when indexing cannot start due to invalid configuration."""


class IndexingProviderError(IndexingError):
    """Raised when the provider fails or returns an invalid response."""


class EmptyIndexResultError(IndexingProviderError):
    """Raised when the provider returns no parsed structured output."""


class Indexer(Protocol):
    """Generate a structured episode index from a normalized transcript."""

    def index(self, transcript: Transcript, prompt: str) -> IndexResult:
        """Index ``transcript`` according to the fully rendered ``prompt``."""
        ...
