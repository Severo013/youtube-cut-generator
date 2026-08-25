"""Command-line entry point for the podcast indexer."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Settings, load_settings
from src.indexing import IndexingError
from src.indexing.openai_indexer import OpenAIIndexer
from src.pipeline import PodcastPipeline
from src.transcription import TranscriptionError
from src.transcription.openai_transcriber import OpenAITranscriber


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
        description="Transcreve e indexa localmente episódios de podcast.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = commands.add_parser(
        "transcribe",
        help="Executa somente a etapa de transcrição",
    )
    transcribe_parser.add_argument("audio_path", type=Path)
    _add_output_argument(transcribe_parser)

    process_parser = commands.add_parser(
        "process",
        help="Executa transcrição e indexação",
    )
    process_parser.add_argument("audio_path", type=Path)
    _add_output_argument(process_parser)

    index_parser = commands.add_parser(
        "index",
        help="Reindexa um transcript.json sem transcrever novamente",
    )
    index_parser.add_argument("transcript_path", type=Path)
    return parser


def run_transcription(audio_path: Path, output_root: Path, settings: Settings) -> Path:
    """Execute only Speech-to-Text and return the episode output directory."""

    episode_output_dir = output_root / audio_path.stem
    transcriber = OpenAITranscriber(
        api_key=settings.openai_api_key,
        model=settings.transcription_model,
        language=settings.transcription_language,
        output_dir=episode_output_dir,
    )
    print("Validando arquivo")
    print("Transcrevendo")
    transcriber.transcribe(audio_path)
    print("Salvando transcrição")
    print("Concluído")
    return episode_output_dir


def run_process(audio_path: Path, output_root: Path, settings: Settings) -> Path:
    """Execute the complete audio-to-index pipeline."""

    episode_output_dir = output_root / audio_path.stem
    transcriber = OpenAITranscriber(
        api_key=settings.openai_api_key,
        model=settings.transcription_model,
        language=settings.transcription_language,
        output_dir=episode_output_dir,
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
        progress=print,
    )
    pipeline.run(audio_path)
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
            output_dir = run_transcription(args.audio_path, args.output_dir, settings)
        elif args.command == "process":
            output_dir = run_process(args.audio_path, args.output_dir, settings)
        else:
            output_dir = run_indexing(args.transcript_path, settings)
    except (FileNotFoundError, IndexingError, TranscriptionError, OSError, ValueError) as error:
        print(f"Erro: {error}")
        return 1

    print(f"Arquivos salvos em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
