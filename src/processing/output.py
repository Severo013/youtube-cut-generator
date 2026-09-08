"""Centralized serialization of task-specific processing results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from src.processing.models import ProcessingResult, TaskDefinition
from src.processing.rendering import (
    MarkdownRenderer,
    build_markdown_renderers,
    render_generic_markdown,
    render_index_markdown,
    render_processing_result,
    render_shorts_markdown,
)


LEGACY_OUTPUT_ALIASES: Mapping[str, tuple[str, str]] = {
    "indexacao": ("index.json", "index.md"),
}


class ProcessingOutputWriter:
    """Write independent task artifacts without coupling executors to files."""

    def __init__(
        self,
        renderers: Mapping[str, MarkdownRenderer] | None = None,
    ) -> None:
        self._renderers = build_markdown_renderers(renderers)

    def write(
        self,
        episode_dir: Path,
        task: TaskDefinition,
        result: ProcessingResult[BaseModel],
    ) -> None:
        task_output_dir = Path(episode_dir) / "processing" / task.id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        markdown = render_processing_result(
            result,
            renderers=self._renderers,
        )
        self._write_json(task_output_dir / "result.json", result.output)
        (task_output_dir / "result.md").write_text(markdown, encoding="utf-8")

        legacy_aliases = LEGACY_OUTPUT_ALIASES.get(task.id)
        if legacy_aliases is not None:
            json_name, markdown_name = legacy_aliases
            self._write_json(Path(episode_dir) / json_name, result.output)
            (Path(episode_dir) / markdown_name).write_text(
                markdown,
                encoding="utf-8",
            )

    @staticmethod
    def _write_json(path: Path, output: BaseModel) -> None:
        payload = output.model_dump(mode="json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
