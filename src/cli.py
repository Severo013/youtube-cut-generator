"""Command-line entry point for the podcast indexer."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_settings
from src.indexing.openai_indexer import OpenAIIndexer
from src.pipeline import PodcastPipeline
from src.transcription.openai_transcriber import OpenAITranscriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcast-indexer",
        description="Transcreve e indexa localmente um episódio de podcast.",
    )
    parser.add_argument("audio_path", type=Path, help="Caminho do arquivo de áudio")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Diretório dos resultados (padrão: outputs)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = load_settings()
    project_dir = Path(__file__).resolve().parent

    pipeline = PodcastPipeline(
        transcriber=OpenAITranscriber(
            api_key=settings.openai_api_key,
            model=settings.transcription_model,
            language=settings.transcription_language,
        ),
        indexer=OpenAIIndexer(
            api_key=settings.openai_api_key,
            model=settings.index_model,
        ),
        prompt_path=project_dir / "prompts" / "decupagem.md",
        output_dir=args.output_dir,
    )

    try:
        result = pipeline.run(args.audio_path)
    except (FileNotFoundError, NotImplementedError, ValueError) as error:
        print(f"Erro: {error}")
        return 1

    print(f"Indexação concluída para: {result.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

