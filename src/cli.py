"""Command-line entry point for the podcast indexer."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from src.config import Settings, load_settings
from src.indexing import IndexingError
from src.input_resolution import InputResolutionError, InputResolver, ResolvedInput
from src.pipeline import PodcastPipeline
from src.processing import (
    ProcessingError,
    ProcessingRunner,
    TaskDefinition,
    TaskRegistry,
)
from src.processing.openai_processor import OpenAITranscriptProcessor
from src.transcription import TranscriptionError
from src.transcription.factory import create_transcriber


DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_TASK_IDS = ("indexacao",)
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parent / "tasks" / "indexacao" / "prompt.md"
)
_INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _safe_directory_name(value: str, fallback: str) -> str:
    """Return a readable, cross-platform directory name."""

    name = _INVALID_FILENAME_CHARACTERS.sub("_", value).strip().rstrip(". ")
    name = name[:200].rstrip(". ")
    if not name:
        return fallback
    if name.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        name = f"_{name}"
    return name


def _episode_output_name(resolved: ResolvedInput) -> str:
    if resolved.kind == "url" and resolved.metadata is not None:
        title = resolved.metadata.title
        if title:
            return _safe_directory_name(title, resolved.path.stem)
    return resolved.path.stem


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório raiz dos resultados (padrão: outputs)",
    )


def _add_tasks_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK_ID",
        help="Uma ou mais tasks (padrão: indexacao)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcast-indexer",
        description="Transcreve e indexa episódios de podcast.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = commands.add_parser(
        "transcribe",
        help="Executa somente a etapa de transcrição",
    )
    transcribe_parser.add_argument(
        "audio_source",
        help="Caminho de arquivo local ou URL HTTP/HTTPS",
    )
    _add_output_argument(transcribe_parser)

    process_parser = commands.add_parser(
        "process",
        help="Executa transcrição e indexação",
    )
    process_parser.add_argument(
        "audio_source",
        help="Caminho de arquivo local ou URL HTTP/HTTPS",
    )
    _add_output_argument(process_parser)
    _add_tasks_argument(process_parser)

    process_transcript_parser = commands.add_parser(
        "process-transcript",
        help="Executa tasks sobre um transcript.json existente",
    )
    process_transcript_parser.add_argument("transcript_path", type=Path)
    _add_tasks_argument(process_transcript_parser)

    index_parser = commands.add_parser(
        "index",
        help="Reindexa um transcript.json sem transcrever novamente",
    )
    index_parser.add_argument("transcript_path", type=Path)
    commands.add_parser("list-tasks", help="Lista as tasks disponíveis")
    return parser


def _transcription_progress(settings: Settings) -> Callable[[str], None]:
    provider = settings.transcription_provider.strip().lower()

    def report(message: str) -> None:
        if message == "Transcrevendo":
            if provider == "local":
                print("Transcrevendo com faster-whisper...")
            elif provider == "openai":
                print("Transcrevendo com OpenAI Whisper...")
            else:
                print(message)
        else:
            print(message)

    return report


def _load_selected_tasks(
    task_ids: Sequence[str] | None,
) -> tuple[TaskRegistry, list[TaskDefinition]]:
    task_registry = TaskRegistry.load_builtin()
    selected_tasks = task_registry.resolve(tuple(task_ids or DEFAULT_TASK_IDS))
    return task_registry, selected_tasks


def _build_runner(settings: Settings) -> ProcessingRunner:
    openai_processor = OpenAITranscriptProcessor(
        api_key=settings.openai_api_key,
        default_model=settings.index_model,
    )
    return ProcessingRunner(executors={"openai": openai_processor})


def run_transcription(
    audio_source: str | Path,
    output_root: Path,
    settings: Settings,
    *,
    resolver: InputResolver | None = None,
) -> Path:
    """Execute only Speech-to-Text and return the episode output directory."""

    resolver = resolver or InputResolver(settings=settings, progress=print)
    with resolver.resolve(audio_source) as resolved:
        episode_output_dir = output_root / _episode_output_name(resolved)
        transcriber = create_transcriber(
            settings=settings,
            output_dir=episode_output_dir,
            progress=print,
        )
        print("Validando arquivo")
        _transcription_progress(settings)("Transcrevendo")
        try:
            transcriber.transcribe(resolved.path)
        finally:
            resolved.cleanup()
        print("Salvando transcrição")
        print("Concluído")
    return episode_output_dir


def run_process(
    audio_source: str | Path,
    output_root: Path,
    settings: Settings,
    *,
    task_ids: Sequence[str] | None = None,
    resolver: InputResolver | None = None,
) -> Path:
    """Transcribe one input and execute the selected transcript tasks."""

    task_registry, selected_tasks = _load_selected_tasks(task_ids)
    runner = _build_runner(settings)
    resolver = resolver or InputResolver(settings=settings, progress=print)
    with resolver.resolve(audio_source) as resolved:
        episode_name = _episode_output_name(resolved)
        episode_output_dir = output_root / episode_name
        transcriber = create_transcriber(
            settings=settings,
            output_dir=episode_output_dir,
            progress=print,
        )
        pipeline = PodcastPipeline.for_processing(
            transcriber=transcriber,
            runner=runner,
            task_registry=task_registry,
            output_dir=output_root,
            progress=_transcription_progress(settings),
        )
        pipeline.run_tasks(
            resolved.path,
            selected_tasks,
            output_name=episode_name,
            after_transcription=resolved.cleanup,
        )
    return episode_output_dir


def run_indexing(transcript_path: Path, settings: Settings) -> Path:
    """Index an existing transcript and return its output directory."""

    return run_process_transcript(
        transcript_path,
        settings,
        task_ids=DEFAULT_TASK_IDS,
    )


def run_process_transcript(
    transcript_path: Path,
    settings: Settings,
    *,
    task_ids: Sequence[str] | None = None,
) -> Path:
    """Execute selected tasks without resolving or transcribing media."""

    task_registry, selected_tasks = _load_selected_tasks(task_ids)
    runner = _build_runner(settings)
    pipeline = PodcastPipeline.for_processing(
        runner=runner,
        task_registry=task_registry,
        output_dir=transcript_path.parent,
        progress=print,
    )
    pipeline.process_transcript_file(transcript_path, selected_tasks)
    return transcript_path.parent


def run_list_tasks(*, registry: TaskRegistry | None = None) -> None:
    """Print tasks discovered by the registry without loading providers."""

    registry = registry or TaskRegistry.load_builtin()
    print("Available tasks:")
    print()
    for task in registry.all():
        print(f"* {task.id} — {task.name}")


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.command == "list-tasks":
            run_list_tasks()
            return 0

        settings = load_settings()
        if args.command == "transcribe":
            output_dir = run_transcription(args.audio_source, args.output_dir, settings)
        elif args.command == "process":
            output_dir = run_process(
                args.audio_source,
                args.output_dir,
                settings,
                task_ids=args.tasks,
            )
        elif args.command == "process-transcript":
            output_dir = run_process_transcript(
                args.transcript_path,
                settings,
                task_ids=args.tasks,
            )
        elif args.command == "index":
            output_dir = run_indexing(args.transcript_path, settings)
        else:  # pragma: no cover - argparse restricts command values
            raise ValueError(f"Comando desconhecido: {args.command}")
    except (
        FileNotFoundError,
        IndexingError,
        InputResolutionError,
        ProcessingError,
        TranscriptionError,
        OSError,
        ValueError,
    ) as error:
        print(f"Erro: {error}")
        return 1

    print(f"Arquivos salvos em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
