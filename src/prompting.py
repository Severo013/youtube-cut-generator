"""Prompt loading and rendering."""

import re
from collections.abc import Mapping
from pathlib import Path

from src.models import Transcript


TRANSCRIPT_PLACEHOLDER = "{{TRANSCRIPT}}"
_PLACEHOLDER_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")


def load_prompt(path: Path) -> str:
    """Load a UTF-8 prompt template from disk."""

    return path.read_text(encoding="utf-8")


def render_prompt(
    template: str,
    variables: Mapping[str, str] | str,
) -> str:
    """Replace known prompt variables once over the original template."""

    if TRANSCRIPT_PLACEHOLDER not in template:
        raise ValueError(f"prompt template must contain {TRANSCRIPT_PLACEHOLDER}")

    resolved_variables = (
        {"TRANSCRIPT": variables}
        if isinstance(variables, str)
        else dict(variables)
    )
    placeholders = _PLACEHOLDER_PATTERN.findall(template)
    unknown = sorted(set(placeholders) - resolved_variables.keys())
    if unknown:
        names = ", ".join(f"{{{{{name}}}}}" for name in unknown)
        raise ValueError(f"unknown prompt placeholder(s): {names}")

    return _PLACEHOLDER_PATTERN.sub(
        lambda match: resolved_variables[match.group(1)],
        template,
    )


def render_decupagem_prompt(template: str, transcript: Transcript) -> str:
    """Compatibility wrapper for the original indexing prompt renderer."""

    return render_prompt(template, transcript.as_timestamped_text())
