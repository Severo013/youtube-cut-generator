"""Local faster-whisper Speech-to-Text adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from src.models import Transcript, TranscriptSegment
from src.transcription.base import (
    EmptyTranscriptionError,
    EmptyTranscriptTextError,
    InvalidAudioFileError,
    TranscriptionProviderError,
    UnsupportedAudioFormatError,
)


SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {
        ".flac",
        ".m4a",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
    }
)

ModelFactory = Callable[..., Any]
ProgressCallback = Callable[[str], None]


class FasterWhisperTranscriber:
    """Transcribe locally with one reusable faster-whisper model instance."""

    def __init__(
        self,
        model: str = "turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "pt",
        output_dir: Path | None = None,
        progress: ProgressCallback | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._output_dir = output_dir
        self._progress = progress or (lambda _message: None)
        self._model_factory = model_factory
        self._model: Any | None = None

    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe an audio file and normalize all lazy segments."""

        audio_path = Path(audio_path)
        self._validate_audio_file(audio_path)
        model = self._get_model()

        self._progress("Transcrevendo...")
        started_at = perf_counter()
        try:
            segment_generator, info = model.transcribe(
                str(audio_path),
                language=self._language,
                word_timestamps=False,
            )
            raw_segments = list(segment_generator)
        except Exception as error:
            if self._device.strip().lower() == "cuda":
                raise TranscriptionProviderError(
                    "A transcrição local não pôde usar a GPU CUDA. "
                    "Verifique o driver NVIDIA e as DLLs CUDA/cuBLAS/cuDNN. "
                    "Nenhum fallback para CPU foi realizado."
                ) from error
            raise TranscriptionProviderError(
                f"A transcrição local falhou no device {self._device!r}."
            ) from error
        elapsed = perf_counter() - started_at

        transcript = self.normalize_segments(
            segments=raw_segments,
            info=info,
            source=str(audio_path),
            fallback_language=self._language,
        )
        if self._output_dir is not None:
            self._save_artifacts(
                transcript=transcript,
                info=info,
                raw_segments=raw_segments,
                elapsed=elapsed,
                output_dir=self._output_dir,
            )
        self._progress(f"Transcrição concluída em {elapsed:.2f} segundos.")
        return transcript

    @classmethod
    def normalize_segments(
        cls,
        segments: list[object],
        info: object,
        source: str,
        fallback_language: str,
    ) -> Transcript:
        """Convert fully consumed faster-whisper segments to ``Transcript``."""

        if not segments:
            raise EmptyTranscriptionError(
                "O faster-whisper não retornou segmentos com timestamps."
            )

        try:
            normalized_segments = [
                TranscriptSegment(
                    start_seconds=cls._get_field(segment, "start"),
                    end_seconds=cls._get_field(segment, "end"),
                    text=cls._get_field(segment, "text"),
                )
                for segment in segments
            ]
            full_text = " ".join(segment.text for segment in normalized_segments)
            if not full_text.strip():
                raise EmptyTranscriptTextError(
                    "O faster-whisper não retornou texto transcrito."
                )

            reported_duration = cls._get_field(info, "duration", default=None)
            duration = (
                float(reported_duration)
                if reported_duration is not None
                else max(segment.end_seconds for segment in normalized_segments)
            )
            language = cls._get_field(info, "language", default=None)
            return Transcript(
                source=source,
                duration_seconds=duration,
                language=str(language or fallback_language),
                full_text=full_text,
                segments=normalized_segments,
            )
        except EmptyTranscriptTextError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError, ValidationError) as error:
            raise TranscriptionProviderError(
                "O faster-whisper retornou uma transcrição em formato inválido."
            ) from error

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        self._progress("Carregando Whisper local...")
        self._progress(f"Modelo: {self._model_name}")
        self._progress(f"Device: {self._device}")
        self._progress(f"Compute: {self._compute_type}")

        if self._device.strip().lower() == "cuda":
            self._validate_cuda()

        try:
            if self._model_factory is None:
                from faster_whisper import WhisperModel

                model_factory: ModelFactory = WhisperModel
            else:
                model_factory = self._model_factory
            model = model_factory(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        except Exception as error:
            message = (
                "A transcrição local não pôde carregar o modelo na GPU CUDA. "
                "Verifique o driver NVIDIA e as DLLs CUDA/cuBLAS/cuDNN. "
                "Nenhum fallback para CPU foi realizado."
                if self._device.strip().lower() == "cuda"
                else f"Não foi possível carregar o modelo no device {self._device!r}."
            )
            raise TranscriptionProviderError(message) from error

        backend = getattr(model, "model", None)
        actual_device = getattr(backend, "device", self._device)
        actual_compute_type = getattr(backend, "compute_type", self._compute_type)
        if (
            str(actual_device).lower() != self._device.strip().lower()
            or str(actual_compute_type).lower() != self._compute_type.strip().lower()
        ):
            raise TranscriptionProviderError(
                "O faster-whisper carregou com configuração diferente da solicitada "
                f"(device={actual_device}, compute_type={actual_compute_type}). "
                "Nenhum fallback silencioso foi aceito."
            )

        self._model = model
        return model

    def _validate_cuda(self) -> None:
        try:
            import ctranslate2

            device_count = ctranslate2.get_cuda_device_count()
            compute_types = ctranslate2.get_supported_compute_types("cuda")
        except Exception as error:
            raise TranscriptionProviderError(
                "A transcrição local não pôde consultar a GPU CUDA. "
                "Verifique o driver NVIDIA e as DLLs CUDA/cuBLAS/cuDNN."
            ) from error
        if device_count < 1:
            raise TranscriptionProviderError(
                "A transcrição local não encontrou uma GPU CUDA. "
                "Nenhum fallback para CPU foi realizado."
            )
        if self._compute_type not in compute_types:
            available = ", ".join(sorted(compute_types))
            raise TranscriptionProviderError(
                f"A GPU CUDA não suporta compute_type={self._compute_type!r}. "
                f"Tipos disponíveis: {available}. Nenhum fallback para CPU foi realizado."
            )

    @staticmethod
    def _validate_audio_file(audio_path: Path) -> None:
        if not audio_path.exists():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")
        if not audio_path.is_file():
            raise InvalidAudioFileError(f"O caminho não é um arquivo: {audio_path}")
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise UnsupportedAudioFormatError(
                f"Extensão de áudio não suportada: "
                f"{audio_path.suffix or '(sem extensão)'}. Formatos aceitos: {supported}."
            )
        try:
            if audio_path.stat().st_size == 0:
                raise InvalidAudioFileError(
                    f"O arquivo de áudio está vazio: {audio_path}"
                )
        except OSError as error:
            raise InvalidAudioFileError(
                f"Não foi possível acessar o arquivo de áudio: {audio_path}"
            ) from error

    def _save_artifacts(
        self,
        transcript: Transcript,
        info: object,
        raw_segments: list[object],
        elapsed: float,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_payload = {
            "provider": "faster-whisper",
            "model": self._model_name,
            "device": self._device,
            "compute_type": self._compute_type,
            "language": self._get_field(info, "language", default=self._language),
            "language_probability": self._get_field(
                info, "language_probability", default=None
            ),
            "duration": self._get_field(info, "duration", default=None),
            "duration_after_vad": self._get_field(
                info, "duration_after_vad", default=None
            ),
            "transcription_seconds": elapsed,
            "segments": [self._serialize_segment(segment) for segment in raw_segments],
        }
        self._write_json(output_dir / "transcript.raw.json", raw_payload)
        self._write_json(
            output_dir / "transcript.json",
            transcript.model_dump(mode="json"),
        )
        timestamped_text = transcript.as_timestamped_text()
        (output_dir / "transcript.txt").write_text(
            f"{timestamped_text}\n" if timestamped_text else "",
            encoding="utf-8",
        )

    @classmethod
    def _serialize_segment(cls, segment: object) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "start": cls._get_field(segment, "start"),
            "end": cls._get_field(segment, "end"),
            "text": cls._get_field(segment, "text"),
        }
        for field in (
            "id",
            "seek",
            "tokens",
            "temperature",
            "avg_logprob",
            "compression_ratio",
            "no_speech_prob",
        ):
            value = cls._get_field(segment, field, default=None)
            if value is not None:
                payload[field] = value
        return payload

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _get_field(value: object, name: str, default: Any = ...) -> Any:
        if isinstance(value, Mapping):
            if default is ...:
                return value[name]
            return value.get(name, default)
        if default is ...:
            return getattr(value, name)
        return getattr(value, name, default)
