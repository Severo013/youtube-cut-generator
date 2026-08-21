"""Command-line entry point for the podcast indexer."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Settings, load_settings
from src.transcription import TranscriptionError
from src.transcription.openai_transcriber import OpenAITranscriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcast-indexer",
        description="Processa localmente episódios de podcast.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    transcribe_parser = commands.add_parser(
        "transcribe",
        help="Executa somente a etapa de transcrição",
    )
    transcribe_parser.add_argument(
        "audio_path",
        type=Path,
        help="Caminho do arquivo de áudio",
    )
    transcribe_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Diretório raiz dos resultados (padrão: outputs)",
    )
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
    transcriber.transcribe(audio_path)
    return episode_output_dir


def main() -> int:
    args = build_parser().parse_args()
    settings = load_settings()

    try:
        output_dir = run_transcription(args.audio_path, args.output_dir, settings)
    except (FileNotFoundError, TranscriptionError, OSError) as error:
        print(f"Erro: {error}")
        return 1

    print(f"Transcrição concluída. Arquivos salvos em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
