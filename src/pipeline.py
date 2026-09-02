"""Transcription orchestration with compatible transcript processing outputs."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from src.indexing.base import Indexer
from src.models import IndexResult, Transcript
from src.processing import (
    ProcessingResult,
    ProcessingRunner,
    TaskDefinition,
    TaskRegistry,
)
from src.processing.output import ProcessingOutputWriter
from src.prompting import load_prompt, render_prompt
from src.transcription.base import Transcriber


ProgressCallback = Callable[[str], None]
DEFAULT_PROCESSING_TASK_ID = "indexacao"


class _LegacyIndexerProcessor:
    """Temporary adapter from the original Indexer contract to the new runner."""

    def __init__(self, indexer: Indexer, prompt_path: Path | None) -> None:
        self._indexer = indexer
        self._prompt_path = prompt_path

    def process(
        self,
        task: TaskDefinition,
        transcript: Transcript,
        formatted_transcript: str,
    ) -> IndexResult:
        template = (
            load_prompt(self._prompt_path)
            if self._prompt_path is not None
            else task.load_prompt()
        )
        prompt = render_prompt(template, formatted_transcript)
        return self._indexer.index(transcript, prompt)

    def model_for(self, task: TaskDefinition) -> str | None:
        return task.model


class PodcastPipeline:
    """Coordinate providers and persist their normalized results locally."""

    def __init__(
        self,
        indexer: Indexer | None,
        prompt_path: Path | None,
        output_dir: Path,
        transcriber: Transcriber | None = None,
        progress: ProgressCallback | None = None,
        *,
        runner: ProcessingRunner | None = None,
        task_registry: TaskRegistry | None = None,
        output_writer: ProcessingOutputWriter | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._output_dir = output_dir
        self._progress = progress or (lambda _message: None)
        self._task_registry = task_registry or TaskRegistry.load_builtin()
        self._output_writer = output_writer or ProcessingOutputWriter()
        self._supports_multiple_tasks = runner is not None

        if runner is not None and indexer is not None:
            raise ValueError("Informe runner ou indexer legado, não ambos.")
        if runner is not None:
            self._runner = runner
        elif indexer is not None:
            task = self._task_registry.get(DEFAULT_PROCESSING_TASK_ID)
            self._runner = ProcessingRunner(
                {task.executor: _LegacyIndexerProcessor(indexer, prompt_path)}
            )
        else:
            raise ValueError("O pipeline requer um ProcessingRunner ou Indexer.")

    @classmethod
    def for_processing(
        cls,
        *,
        runner: ProcessingRunner,
        task_registry: TaskRegistry,
        output_dir: Path,
        transcriber: Transcriber | None = None,
        progress: ProgressCallback | None = None,
        output_writer: ProcessingOutputWriter | None = None,
    ) -> "PodcastPipeline":
        """Build the pipeline through the generic processing architecture."""

        return cls(
            indexer=None,
            prompt_path=None,
            output_dir=output_dir,
            transcriber=transcriber,
            progress=progress,
            runner=runner,
            task_registry=task_registry,
            output_writer=output_writer,
        )

    def run(
        self,
        audio_path: Path,
        *,
        output_name: str | None = None,
        after_transcription: Callable[[], None] | None = None,
    ) -> IndexResult:
        """Run transcription and indexing for one audio file.

        ``after_transcription`` releases short-lived input resources before the
        provider-independent indexing stage starts.
        """

        task = self._task_registry.get(DEFAULT_PROCESSING_TASK_ID)
        results = self._run_audio_tasks(
            audio_path,
            [task],
            output_name=output_name,
            after_transcription=after_transcription,
            legacy_progress=True,
        )
        return IndexResult.model_validate(results[0].output)

    def run_tasks(
        self,
        audio_path: Path,
        tasks: Sequence[TaskDefinition],
        *,
        output_name: str | None = None,
        after_transcription: Callable[[], None] | None = None,
    ) -> list[ProcessingResult]:
        """Transcribe once and execute selected independent tasks sequentially."""

        self._require_generic_processing()
        return self._run_audio_tasks(
            audio_path,
            tasks,
            output_name=output_name,
            after_transcription=after_transcription,
            legacy_progress=False,
        )

    def index_transcript_file(self, transcript_path: Path) -> IndexResult:
        """Index an existing normalized transcript without transcribing again."""

        transcript_path, transcript = self._load_transcript(transcript_path)
        self._progress("Indexando episódio")
        task = self._task_registry.get(DEFAULT_PROCESSING_TASK_ID)
        results = self._process_tasks(transcript_path.parent, transcript, [task])

        self._progress("Salvando resultado")
        self._progress("Concluído")
        return IndexResult.model_validate(results[0].output)

    def process_transcript_file(
        self,
        transcript_path: Path,
        tasks: Sequence[TaskDefinition],
    ) -> list[ProcessingResult]:
        """Execute selected tasks over an existing normalized transcript."""

        self._require_generic_processing()
        transcript_path, transcript = self._load_transcript(transcript_path)
        self._progress("Processando tasks")
        results = self._process_tasks(transcript_path.parent, transcript, tasks)
        self._progress("Concluído")
        return results

    def _create_index(self, transcript: Transcript) -> IndexResult:
        task = self._task_registry.get(DEFAULT_PROCESSING_TASK_ID)
        processing_results = self._runner.run(transcript, [task])
        if len(processing_results) != 1:
            raise RuntimeError("A task de indexação não retornou exatamente um resultado.")
        return IndexResult.model_validate(processing_results[0].output)

    def _run_audio_tasks(
        self,
        audio_path: Path,
        tasks: Sequence[TaskDefinition],
        *,
        output_name: str | None,
        after_transcription: Callable[[], None] | None,
        legacy_progress: bool,
    ) -> list[ProcessingResult]:
        if not tasks:
            raise ValueError("Selecione ao menos uma task.")

        audio_path = Path(audio_path)
        self._progress("Validando arquivo")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")
        if self._transcriber is None:
            raise RuntimeError("O pipeline completo requer um Transcriber.")

        episode_dir = self._output_dir / (output_name or audio_path.stem)
        self._progress("Transcrevendo")
        try:
            transcript = self._transcriber.transcribe(audio_path)
        finally:
            if after_transcription is not None:
                after_transcription()

        self._progress("Salvando transcrição")
        self._save_transcript(episode_dir, transcript)
        self._progress("Indexando episódio" if legacy_progress else "Processando tasks")
        results = self._process_tasks(episode_dir, transcript, tasks)
        if legacy_progress:
            self._progress("Salvando resultado")
        self._progress("Concluído")
        return results

    def _process_tasks(
        self,
        episode_dir: Path,
        transcript: Transcript,
        tasks: Sequence[TaskDefinition],
    ) -> list[ProcessingResult]:
        if not tasks:
            raise ValueError("Selecione ao menos uma task.")

        def save_result(
            task: TaskDefinition,
            result: ProcessingResult,
        ) -> None:
            self._output_writer.write(episode_dir, task, result)

        return self._runner.run(transcript, tasks, on_result=save_result)

    def _load_transcript(self, transcript_path: Path) -> tuple[Path, Transcript]:
        transcript_path = Path(transcript_path)
        self._progress("Validando transcrição")
        if not transcript_path.is_file():
            raise FileNotFoundError(
                f"Arquivo de transcrição não encontrado: {transcript_path}"
            )
        transcript = Transcript.model_validate_json(
            transcript_path.read_text(encoding="utf-8")
        )
        return transcript_path, transcript

    def _require_generic_processing(self) -> None:
        if not self._supports_multiple_tasks:
            raise RuntimeError(
                "O adapter legado de Indexer não suporta múltiplas tasks."
            )

    @classmethod
    def _save_transcript(cls, episode_dir: Path, transcript: Transcript) -> None:
        episode_dir.mkdir(parents=True, exist_ok=True)
        cls._write_json(episode_dir / "transcript.json", transcript)
        timestamped_text = transcript.as_timestamped_text()
        (episode_dir / "transcript.txt").write_text(
            f"{timestamped_text}\n" if timestamped_text else "",
            encoding="utf-8",
        )

    @staticmethod
    def _write_json(path: Path, model: Transcript) -> None:
        payload = model.model_dump(mode="json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
