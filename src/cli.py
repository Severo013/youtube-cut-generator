"""Command-line entry point for the podcast indexer."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from pathlib import Path

from src.config import Settings, load_settings
from src.indexing import IndexingError
from src.input_resolution import InputResolutionError, InputResolver, ResolvedInput
from src.pipeline import PodcastPipeline
from src.processing import ProcessingError, ProcessingRunner, TaskRegistry
from src.processing.openai_processor import OpenAITranscriptProcessor
from src.transcription import TranscriptionError
from src.transcription.factory import create_transcriber


DEFAULT_OUTPUT_DIR = Path("outputs")
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

    index_parser = commands.add_parser(
        "index",
        help="Reindexa um transcript.json sem transcrever novamente",
    )
    index_parser.add_argument("transcript_path", type=Path)
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


def _build_processing(
    settings: Settings,
) -> tuple[TaskRegistry, ProcessingRunner]:
    task_registry = TaskRegistry.load_builtin()
    openai_processor = OpenAITranscriptProcessor(
        api_key=settings.openai_api_key,
        default_model=settings.index_model,
    )
    runner = ProcessingRunner({"openai": openai_processor})
    return task_registry, runner


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
    resolver: InputResolver | None = None,
) -> Path:
    """Execute the complete audio-to-index pipeline."""

    resolver = resolver or InputResolver(settings=settings, progress=print)
    with resolver.resolve(audio_source) as resolved:
        episode_name = _episode_output_name(resolved)
        episode_output_dir = output_root / episode_name
        transcriber = create_transcriber(
            settings=settings,
            output_dir=episode_output_dir,
            progress=print,
        )
        task_registry, runner = _build_processing(settings)
        pipeline = PodcastPipeline.for_processing(
            transcriber=transcriber,
            runner=runner,
            task_registry=task_registry,
            output_dir=output_root,
            progress=_transcription_progress(settings),
        )
        pipeline.run(
            resolved.path,
            output_name=episode_name,
            after_transcription=resolved.cleanup,
        )
    return episode_output_dir


def run_indexing(transcript_path: Path, settings: Settings) -> Path:
    """Index an existing transcript and return its output directory."""

    task_registry, runner = _build_processing(settings)
    pipeline = PodcastPipeline.for_processing(
        runner=runner,
        task_registry=task_registry,
        output_dir=transcript_path.parent,
        progress=print,
    )
    pipeline.index_transcript_file(transcript_path)
    return transcript_path.parent


def main() -> int:
    args = build_parser().parse_args()
    settings = load_settings()

    try:
        if args.command == "transcribe":
            output_dir = run_transcription(args.audio_source, args.output_dir, settings)
        elif args.command == "process":
            output_dir = run_process(args.audio_source, args.output_dir, settings)
        else:
            output_dir = run_indexing(args.transcript_path, settings)
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
