"""Validate faster-whisper with CUDA without changing the application pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import NoReturn

MODEL = "turbo"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE = "pt"
CACHE_ENV_VAR = "FASTER_WHISPER_CACHE_DIR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testa o faster-whisper turbo em CUDA/FP16, sem fallback para CPU."
    )
    parser.add_argument("audio", type=Path, help="Caminho do arquivo de áudio")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Diretório opcional para o cache/download do modelo. "
            f"Também pode ser definido por {CACHE_ENV_VAR}."
        ),
    )
    return parser.parse_args()


def fail(message: str, error: BaseException | None = None) -> NoReturn:
    print(f"ERRO: {message}", file=sys.stderr)
    if error is not None:
        print(f"Detalhes: {type(error).__name__}: {error}", file=sys.stderr)
    raise SystemExit(1)


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_gpu_name() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "GPU CUDA 0 (nome indisponível via nvidia-smi)"
    return result.stdout.strip() or "GPU CUDA 0 (nome não informado)"


def main() -> None:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        fail(f"arquivo de áudio não encontrado: {audio_path}")

    try:
        import ctranslate2
        import faster_whisper
        from faster_whisper import WhisperModel
    except (ImportError, OSError) as error:
        fail(
            "faster-whisper/CTranslate2 não está disponível neste ambiente. "
            "Ative a .venv e instale o projeto com `python -m pip install -e \".[dev]\"`.",
            error,
        )

    try:
        cuda_device_count = ctranslate2.get_cuda_device_count()
    except Exception as error:
        fail(
            "o CTranslate2 não conseguiu consultar dispositivos CUDA. "
            "Verifique o driver NVIDIA e as DLLs CUDA exigidas no PATH.",
            error,
        )
    if cuda_device_count < 1:
        fail(
            "nenhuma GPU CUDA foi reconhecida pelo CTranslate2; "
            "o teste foi interrompido antes da transcrição (sem fallback para CPU)."
        )

    try:
        supported_compute_types = ctranslate2.get_supported_compute_types(DEVICE)
    except Exception as error:
        fail("não foi possível consultar os compute types da GPU CUDA.", error)
    if COMPUTE_TYPE not in supported_compute_types:
        fail(
            f"a GPU não anunciou suporte a {COMPUTE_TYPE}; tipos disponíveis: "
            f"{', '.join(sorted(supported_compute_types))}."
        )

    configured_cache = args.cache_dir or (
        Path(cache_value) if (cache_value := os.getenv(CACHE_ENV_VAR)) else None
    )
    download_root = (
        str(configured_cache.expanduser().resolve()) if configured_cache else None
    )

    print(f"faster-whisper: {faster_whisper.__version__}")
    print(f"CTranslate2: {ctranslate2.__version__}")
    print(f"GPUs CUDA reconhecidas: {cuda_device_count}")
    print(f"GPU utilizada: {get_gpu_name()}")
    print(f"Modelo: {MODEL}")
    print(f"Device solicitado: {DEVICE}")
    print(f"Compute type: {COMPUTE_TYPE}")
    if download_root:
        print(f"Cache do modelo: {download_root}")

    try:
        model = WhisperModel(
            MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=download_root,
        )
    except Exception as error:
        fail(
            "não foi possível carregar o modelo turbo em CUDA/float16. "
            "Nenhum fallback para CPU foi realizado. No Windows, confirme que as "
            "DLLs CUDA compatíveis estão instaladas e acessíveis pelo PATH.",
            error,
        )
    if model.model.device != DEVICE or model.model.compute_type != COMPUTE_TYPE:
        fail(
            "o backend carregou com configuração diferente da solicitada "
            f"(device={model.model.device}, compute_type={model.model.compute_type}); "
            "o teste foi interrompido para impedir fallback silencioso."
        )

    started_at = perf_counter()
    try:
        segment_generator, info = model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            word_timestamps=False,
        )
        segments = list(segment_generator)
    except Exception as error:
        fail(
            "a transcrição CUDA falhou. Nenhuma nova tentativa em CPU foi iniciada.",
            error,
        )
    elapsed = perf_counter() - started_at

    print(f"Idioma detectado: {info.language}")
    language_probability = getattr(info, "language_probability", None)
    if language_probability is not None:
        print(f"Probabilidade do idioma: {language_probability:.4f}")
    print(f"Duração aproximada: {format_timestamp(info.duration)} ({info.duration:.2f} s)")
    print(f"Tempo de transcrição: {elapsed:.2f} s")
    print(f"Quantidade de segmentos: {len(segments)}")
    if elapsed > 0 and info.duration > 0:
        realtime_factor = info.duration / elapsed
        if realtime_factor >= 1:
            print(
                f"Razão de processamento: {realtime_factor:.2f}x mais rápido "
                "que tempo real"
            )
        else:
            print(
                f"Razão de processamento: {1 / realtime_factor:.2f}x mais lento "
                "que tempo real"
            )

    for segment in segments:
        text = segment.text.strip()
        print(
            f"[{format_timestamp(segment.start)} - "
            f"{format_timestamp(segment.end)}] {text}"
        )


if __name__ == "__main__":
    main()
