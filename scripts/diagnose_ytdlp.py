"""Diagnose yt-dlp/EJS and optionally download and transcribe one URL."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_settings
from src.input_resolution import (
    InputResolutionError,
    InputResolver,
    inspect_ytdlp_environment,
)
from src.transcription import TranscriptionError
from src.transcription.factory import create_transcriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "url",
        nargs="?",
        help="URL HTTP/HTTPS para o teste completo",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    environment = inspect_ytdlp_environment()
    print(f"yt-dlp: {environment.yt_dlp_version or 'não instalado'}")
    print(f"yt-dlp-ejs: {environment.ejs_version or 'não instalado'}")
    if environment.deno_path:
        status = "suportado" if environment.deno_supported else "desatualizado"
        print(
            f"Runtime JavaScript: Deno {environment.deno_version or 'versão desconhecida'} "
            f"({status}) em {environment.deno_path}"
        )
    else:
        print("Runtime JavaScript: Deno não encontrado no PATH")
    ejs_ready = bool(environment.ejs_version and environment.deno_supported)
    print(f"Suporte EJS pronto: {'sim' if ejs_ready else 'não'}")

    if not args.url:
        return 0

    settings = load_settings()
    resolver = InputResolver(settings=settings, progress=print)
    downloaded_path: Path | None = None
    exit_code = 0
    try:
        with resolver.resolve(args.url) as resolved:
            downloaded_path = resolved.path
            metadata = resolved.metadata
            if metadata is None:
                raise RuntimeError("O diagnóstico esperava metadata de uma URL.")
            print(f"Autenticação utilizada: {metadata.authentication_method}")
            print(
                "Formato de áudio: "
                f"{metadata.extension} "
                f"(format_id={metadata.format_id or 'desconhecido'}, "
                f"acodec={metadata.audio_codec or 'desconhecido'}, "
                f"vcodec={metadata.video_codec or 'desconhecido'})"
            )
            print(f"Tamanho: {metadata.file_size_bytes} bytes")
            print(f"Tempo de download: {metadata.download_seconds:.2f} segundos")
            print(f"Caminho temporário: {resolved.path}")

            transcriber = create_transcriber(settings=settings, progress=print)
            transcript = transcriber.transcribe(resolved.path)
            print(
                "Transcrição: funcionando "
                f"({len(transcript.segments)} segmentos, "
                f"{transcript.duration_seconds:.2f} segundos)"
            )
    except (InputResolutionError, TranscriptionError, OSError, ValueError) as error:
        print(f"Erro: {error}")
        exit_code = 1

    if downloaded_path is not None:
        if settings.ytdlp_keep_temp:
            print(
                "Remoção ao final: preservado por configuração "
                f"({downloaded_path})"
            )
        else:
            status = "falhou" if downloaded_path.exists() else "confirmada"
            print(f"Remoção ao final: {status}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
