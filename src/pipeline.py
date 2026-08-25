"""Provider-independent orchestration for transcription and indexing."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from src.indexing.base import Indexer
from src.models import IndexResult, Transcript
from src.prompting import load_prompt, render_decupagem_prompt
from src.transcription.base import Transcriber


ProgressCallback = Callable[[str], None]


class PodcastPipeline:
    """Coordinate providers and persist their normalized results locally."""

    def __init__(
        self,
        indexer: Indexer,
        prompt_path: Path,
        output_dir: Path,
        transcriber: Transcriber | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._indexer = indexer
        self._prompt_path = prompt_path
        self._output_dir = output_dir
        self._progress = progress or (lambda _message: None)

    def run(self, audio_path: Path) -> IndexResult:
        """Run transcription and indexing for one audio file."""

        audio_path = Path(audio_path)
        self._progress("Validando arquivo")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")
        if self._transcriber is None:
            raise RuntimeError("O pipeline completo requer um Transcriber.")

        episode_dir = self._output_dir / audio_path.stem
        self._progress("Transcrevendo")
        transcript = self._transcriber.transcribe(audio_path)

        self._progress("Salvando transcrição")
        self._save_transcript(episode_dir, transcript)

        self._progress("Indexando episódio")
        result = self._create_index(transcript)

        self._progress("Salvando resultado")
        self._save_index(episode_dir, result)
        self._progress("Concluído")
        return result

    def index_transcript_file(self, transcript_path: Path) -> IndexResult:
        """Index an existing normalized transcript without transcribing again."""

        transcript_path = Path(transcript_path)
        self._progress("Validando transcrição")
        if not transcript_path.is_file():
            raise FileNotFoundError(
                f"Arquivo de transcrição não encontrado: {transcript_path}"
            )

        transcript = Transcript.model_validate_json(
            transcript_path.read_text(encoding="utf-8")
        )
        self._progress("Indexando episódio")
        result = self._create_index(transcript)

        self._progress("Salvando resultado")
        self._save_index(transcript_path.parent, result)
        self._progress("Concluído")
        return result

    def _create_index(self, transcript: Transcript) -> IndexResult:
        template = load_prompt(self._prompt_path)
        prompt = render_decupagem_prompt(template, transcript)
        return self._indexer.index(transcript, prompt)

    @classmethod
    def _save_transcript(cls, episode_dir: Path, transcript: Transcript) -> None:
        episode_dir.mkdir(parents=True, exist_ok=True)
        cls._write_json(episode_dir / "transcript.json", transcript)
        timestamped_text = transcript.as_timestamped_text()
        (episode_dir / "transcript.txt").write_text(
            f"{timestamped_text}\n" if timestamped_text else "",
            encoding="utf-8",
        )

    @classmethod
    def _save_index(cls, episode_dir: Path, result: IndexResult) -> None:
        episode_dir.mkdir(parents=True, exist_ok=True)
        cls._write_json(episode_dir / "index.json", result)

        sections = ["# Indexação do episódio"]
        for block in result.blocks:
            sections.extend(
                [
                    "",
                    (
                        f"## {block.start_timestamp} - {block.end_timestamp} "
                        f"— {block.topic}"
                    ),
                    "",
                    block.summary,
                ]
            )
        (episode_dir / "index.md").write_text(
            "\n".join(sections) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_json(path: Path, model: Transcript | IndexResult) -> None:
        payload = model.model_dump(mode="json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
