"""Application orchestration for transcription and indexing."""

from __future__ import annotations

import json
from pathlib import Path

from src.indexing.base import Indexer
from src.models import IndexResult, Transcript
from src.prompting import load_prompt, render_decupagem_prompt
from src.transcription.base import Transcriber


class PodcastPipeline:
    """Coordinate providers and persist their normalized results locally."""

    def __init__(
        self,
        transcriber: Transcriber,
        indexer: Indexer,
        prompt_path: Path,
        output_dir: Path,
    ) -> None:
        self._transcriber = transcriber
        self._indexer = indexer
        self._prompt_path = prompt_path
        self._output_dir = output_dir

    def run(self, audio_path: Path) -> IndexResult:
        """Run the complete pipeline and save transcript and index as JSON."""

        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        transcript = self._transcriber.transcribe(audio_path)
        template = load_prompt(self._prompt_path)
        prompt = render_decupagem_prompt(template, transcript)
        result = self._indexer.index(transcript, prompt)
        self._save_results(audio_path, transcript, result)
        return result

    def _save_results(
        self,
        audio_path: Path,
        transcript: Transcript,
        result: IndexResult,
    ) -> None:
        episode_dir = self._output_dir / audio_path.stem
        episode_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(episode_dir / "transcript.json", transcript)
        self._write_json(episode_dir / "index.json", result)

    @staticmethod
    def _write_json(path: Path, model: Transcript | IndexResult) -> None:
        payload = model.model_dump(mode="json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

