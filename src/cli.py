"""Command-line entry point for the podcast indexer."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from src.config import Settings, load_settings
from src.indexing import IndexingError
from src.indexing.openai_indexer import OpenAIIndexer
from src.input_resolution import InputResolutionError, InputResolver
from src.pipeline import PodcastPipeline
from src.transcription import TranscriptionError
from src.transcription.factory import create_transcriber


DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "decupagem.md"


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
        episode_output_dir = output_root / resolved.path.stem
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
        episode_output_dir = output_root / resolved.path.stem
        transcriber = create_transcriber(
            settings=settings,
            output_dir=episode_output_dir,
            progress=print,
        )
        indexer = OpenAIIndexer(
            api_key=settings.openai_api_key,
            model=settings.index_model,
        )
        pipeline = PodcastPipeline(
            transcriber=transcriber,
            indexer=indexer,
            prompt_path=DEFAULT_PROMPT_PATH,
            output_dir=output_root,
            progress=_transcription_progress(settings),
        )
        pipeline.run(resolved.path, after_transcription=resolved.cleanup)
    return episode_output_dir


def run_indexing(transcript_path: Path, settings: Settings) -> Path:
    """Index an existing transcript and return its output directory."""

    indexer = OpenAIIndexer(
        api_key=settings.openai_api_key,
        model=settings.index_model,
    )
    pipeline = PodcastPipeline(
        indexer=indexer,
        prompt_path=DEFAULT_PROMPT_PATH,
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
