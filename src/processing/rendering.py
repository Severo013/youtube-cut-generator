"""Pure text rendering for structured processing results."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from pydantic import BaseModel

from src.models import IndexResult, seconds_to_timestamp
from src.processing.models import ProcessingResult, TaskDefinition
from src.tasks.cortes.models import CutsResult
from src.tasks.shorts.models import ShortsResult


MarkdownRenderer = Callable[[BaseModel], str]


def render_index_markdown(output: BaseModel) -> str:
    result = IndexResult.model_validate(output)
    sections = ["# Indexação do episódio"]
    for block in result.blocks:
        sections.extend(
            [
                "",
                (
                    f"## {block.start_timestamp} - {block.end_timestamp} "
                    f"— {block.topic}"
                ),
                "",
                block.summary,
            ]
        )
    return "\n".join(sections) + "\n"


def render_shorts_markdown(output: BaseModel) -> str:
    result = ShortsResult.model_validate(output)
    sections = ["# Seleção de Shorts"]
    for position, moment in enumerate(result.moments, start=1):
        title = moment.title or f"Momento {position}"
        sections.extend(
            [
                "",
                (
                    f"## {seconds_to_timestamp(moment.start_seconds)} - "
                    f"{seconds_to_timestamp(moment.end_seconds)} — {title}"
                ),
            ]
        )
        for label, value in (
            ("Resumo", moment.summary),
            ("Ideia central", moment.central_idea),
            ("Contexto", moment.context),
        ):
            if value:
                sections.extend(["", f"**{label}:** {value}"])
    return "\n".join(sections) + "\n"


def render_cuts_markdown(output: BaseModel) -> str:
    result = CutsResult.model_validate(output)
    sections = ["# Seleção de Cortes"]
    for position, cut in enumerate(result.cuts, start=1):
        sections.extend(
            [
                "",
                f"## Corte {position} — {cut.title}",
                "",
                f"**Início:** {seconds_to_timestamp(cut.start_seconds)}",
                "",
                f"**Fim:** {seconds_to_timestamp(cut.end_seconds)}",
                "",
                f"**Duração:** {seconds_to_timestamp(cut.duration_seconds)}",
                "",
                f"**Tema:** {cut.theme}",
                "",
                f"**Resumo:** {cut.summary}",
                "",
                f"**Título:** {cut.title}",
                "",
                f"**Frase para thumbnail:** {cut.thumbnail_text}",
            ]
        )
    if result.limitations:
        sections.extend(["", "## Limitações"])
        sections.extend(f"- {limitation}" for limitation in result.limitations)
    return "\n".join(sections) + "\n"


def _render_generic_markdown(task_name: str, output: BaseModel) -> str:
    payload = json.dumps(
        output.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    return f"# {task_name}\n\n```json\n{payload}\n```\n"


def render_generic_markdown(task: TaskDefinition, output: BaseModel) -> str:
    """Compatibility entry point for rendering a task's generic payload."""

    return _render_generic_markdown(task.name, output)


DEFAULT_MARKDOWN_RENDERERS: Mapping[str, MarkdownRenderer] = {
    "cuts_result": render_cuts_markdown,
    "index_result": render_index_markdown,
    "shorts_result": render_shorts_markdown,
}


def build_markdown_renderers(
    renderers: Mapping[str, MarkdownRenderer] | None = None,
) -> dict[str, MarkdownRenderer]:
    configured = dict(DEFAULT_MARKDOWN_RENDERERS)
    if renderers:
        configured.update(renderers)
    return configured


def render_processing_result(
    result: ProcessingResult[BaseModel],
    *,
    renderers: Mapping[str, MarkdownRenderer] | None = None,
) -> str:
    """Render an in-memory structured result without performing I/O."""

    renderer = renderers.get(result.schema_id) if renderers is not None else None
    if renderer is None:
        renderer = DEFAULT_MARKDOWN_RENDERERS.get(result.schema_id)
    if renderer is not None:
        return renderer(result.output)
    return _render_generic_markdown(result.task_name, result.output)
