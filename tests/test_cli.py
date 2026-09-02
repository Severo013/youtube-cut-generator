import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from src.cli import (
    _safe_directory_name,
    build_parser,
    main,
    run_indexing,
    run_list_tasks,
    run_process,
    run_process_transcript,
    run_transcription,
)
from src.config import Settings
from src.input_resolution import (
    DownloadMetadata,
    DownloadedAudio,
    InputResolver,
    YtDlpDownloader,
)
from src.processing import TaskNotFoundError


class CliTests(unittest.TestCase):
    def test_accepts_all_commands_and_task_selections(self) -> None:
        parser = build_parser()

        process_args = parser.parse_args(["process", "episode.mp3"])
        indexacao_args = parser.parse_args(
            ["process", "episode.mp3", "--tasks", "indexacao"]
        )
        shorts_args = parser.parse_args(
            ["process", "episode.mp3", "--tasks", "shorts"]
        )
        both_args = parser.parse_args(
            ["process", "episode.mp3", "--tasks", "indexacao", "shorts"]
        )
        transcribe_args = parser.parse_args(["transcribe", "episode.mp3"])
        index_args = parser.parse_args(["index", "outputs/episode/transcript.json"])
        transcript_args = parser.parse_args(
            [
                "process-transcript",
                "outputs/episode/transcript.json",
                "--tasks",
                "shorts",
                "indexacao",
            ]
        )
        list_args = parser.parse_args(["list-tasks"])

        self.assertEqual(process_args.command, "process")
        self.assertIsNone(process_args.tasks)
        self.assertEqual(indexacao_args.tasks, ["indexacao"])
        self.assertEqual(shorts_args.tasks, ["shorts"])
        self.assertEqual(both_args.tasks, ["indexacao", "shorts"])
        self.assertEqual(transcribe_args.command, "transcribe")
        self.assertEqual(index_args.command, "index")
        self.assertEqual(transcript_args.command, "process-transcript")
        self.assertEqual(transcript_args.tasks, ["shorts", "indexacao"])
        self.assertEqual(list_args.command, "list-tasks")
        self.assertEqual(process_args.audio_source, "episode.mp3")
        self.assertEqual(index_args.transcript_path.name, "transcript.json")

    @patch("src.cli.create_transcriber")
    def test_transcribe_command_uses_central_provider_factory(self, factory) -> None:
        transcriber = factory.return_value
        settings = Settings(transcription_provider="local")

        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_path = Path(temporary_dir) / "episode.mp3"
            audio_path.write_bytes(b"audio")
            output_dir = run_transcription(audio_path, Path("outputs"), settings)

        factory.assert_called_once_with(
            settings=settings,
            output_dir=Path("outputs/episode"),
            progress=print,
        )
        transcriber.transcribe.assert_called_once_with(audio_path)
        self.assertEqual(output_dir, Path("outputs/episode"))

    @patch("src.cli.create_transcriber")
    def test_url_download_is_passed_to_transcriber_and_then_removed(
        self, factory
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir) / "download"
            audio_path = download_root / "video.webm"
            downloader = Mock()

            def download(url: str, root: Path) -> DownloadedAudio:
                root.mkdir(parents=True)
                audio_path.write_bytes(b"audio")
                metadata = DownloadMetadata(
                    video_id="video",
                    title="Episode: A Test?",
                    webpage_url=url,
                    duration_seconds=1,
                    uploader="Channel",
                    extension="webm",
                    format_id="251",
                    audio_codec="opus",
                    video_codec="none",
                    authentication_method="none",
                    file_size_bytes=5,
                    download_seconds=0.1,
                    temporary_directory=root,
                )
                return DownloadedAudio(audio_path, metadata)

            downloader.download.side_effect = download
            resolver = InputResolver(Settings(), downloader=downloader)
            with patch(
                "src.input_resolution.tempfile.mkdtemp",
                return_value=str(download_root),
            ):
                output_dir = run_transcription(
                    "https://example.com/video",
                    Path("outputs"),
                    Settings(),
                    resolver=resolver,
                )

            factory.return_value.transcribe.assert_called_once_with(audio_path)
            self.assertFalse(download_root.exists())
            self.assertEqual(output_dir, Path("outputs/Episode_ A Test_"))

    def test_sanitizes_url_title_for_a_cross_platform_directory(self) -> None:
        self.assertEqual(
            _safe_directory_name('  CON: an <episode>.  ', "video"),
            "CON_ an _episode_",
        )
        self.assertEqual(_safe_directory_name('  ...  ', "video"), "video")

    @patch("src.cli.PodcastPipeline")
    @patch("src.cli.create_transcriber")
    def test_process_uses_selected_tasks_in_requested_order(
        self,
        create_transcriber,
        pipeline_constructor,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_path = Path(temporary_dir) / "episode.mp3"
            audio_path.write_bytes(b"audio")

            run_process(
                audio_path,
                Path(temporary_dir) / "outputs",
                Settings(),
                task_ids=["shorts", "indexacao"],
            )

        tasks = pipeline_constructor.for_processing.return_value.run_tasks.call_args.args[1]
        self.assertEqual([task.id for task in tasks], ["shorts", "indexacao"])
        create_transcriber.assert_called_once()

    @patch("src.cli.PodcastPipeline")
    @patch("src.cli.create_transcriber")
    def test_process_defaults_to_indexacao(
        self,
        create_transcriber,
        pipeline_constructor,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_path = Path(temporary_dir) / "episode.mp3"
            audio_path.write_bytes(b"audio")

            run_process(
                audio_path,
                Path(temporary_dir) / "outputs",
                Settings(),
            )

        tasks = pipeline_constructor.for_processing.return_value.run_tasks.call_args.args[1]
        self.assertEqual([task.id for task in tasks], ["indexacao"])
        create_transcriber.assert_called_once()

    @patch("src.cli.create_transcriber")
    @patch("src.cli.InputResolver")
    @patch("src.cli.OpenAITranscriptProcessor")
    def test_unknown_task_is_rejected_before_expensive_components(
        self,
        processor_constructor,
        resolver_constructor,
        create_transcriber,
    ) -> None:
        with patch.object(
            YtDlpDownloader,
            "__init__",
            return_value=None,
        ) as downloader_constructor, patch(
            "src.transcription.faster_whisper_transcriber."
            "FasterWhisperTranscriber.__init__",
            return_value=None,
        ) as whisper_constructor:
            with self.assertRaisesRegex(
                TaskNotFoundError,
                "Unknown task: xyz\nAvailable tasks: indexacao, shorts",
            ):
                run_process(
                    "https://example.com/video",
                    Path("outputs"),
                    Settings(),
                    task_ids=["indexacao", "xyz"],
                )

        processor_constructor.assert_not_called()
        resolver_constructor.assert_not_called()
        create_transcriber.assert_not_called()
        downloader_constructor.assert_not_called()
        whisper_constructor.assert_not_called()

    @patch("src.cli.PodcastPipeline")
    @patch("src.cli.create_transcriber")
    @patch("src.cli.InputResolver")
    def test_process_transcript_never_initializes_media_or_transcription(
        self,
        resolver_constructor,
        create_transcriber,
        pipeline_constructor,
    ) -> None:
        transcript_path = Path("outputs/episode/transcript.json")
        pipeline = pipeline_constructor.for_processing.return_value

        with patch.object(
            YtDlpDownloader,
            "__init__",
            return_value=None,
        ) as downloader_constructor, patch(
            "src.transcription.faster_whisper_transcriber."
            "FasterWhisperTranscriber.__init__",
            return_value=None,
        ) as whisper_constructor:
            output_dir = run_process_transcript(
                transcript_path,
                Settings(),
                task_ids=["shorts", "indexacao"],
            )

        tasks = pipeline.process_transcript_file.call_args.args[1]
        self.assertEqual([task.id for task in tasks], ["shorts", "indexacao"])
        resolver_constructor.assert_not_called()
        create_transcriber.assert_not_called()
        downloader_constructor.assert_not_called()
        whisper_constructor.assert_not_called()
        self.assertEqual(output_dir, transcript_path.parent)

    def test_list_tasks_comes_from_registry(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            run_list_tasks()

        self.assertEqual(
            output.getvalue(),
            "Available tasks:\n\n"
            "* indexacao — Indexação do episódio\n"
            "* shorts — Seleção de Shorts\n",
        )

    @patch("src.cli.load_settings")
    def test_list_tasks_main_does_not_load_runtime_settings(self, load_settings) -> None:
        output = io.StringIO()

        with patch("sys.argv", ["podcast-indexer", "list-tasks"]), redirect_stdout(output):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        load_settings.assert_not_called()
        self.assertIn("* shorts — Seleção de Shorts", output.getvalue())

    @patch("src.cli.PodcastPipeline")
    @patch("src.cli.ProcessingRunner")
    @patch("src.cli.OpenAITranscriptProcessor")
    @patch("src.cli.TaskRegistry")
    @patch("src.cli.create_transcriber")
    def test_reindex_command_uses_default_generic_processing_without_transcriber(
        self,
        create_transcriber,
        registry_constructor,
        processor_constructor,
        runner_constructor,
        pipeline_constructor,
    ) -> None:
        transcript_path = Path("outputs/episode/transcript.json")
        settings = Settings()
        registry = registry_constructor.load_builtin.return_value
        selected_tasks = registry.resolve.return_value
        processor = processor_constructor.return_value
        runner = runner_constructor.return_value
        pipeline = pipeline_constructor.for_processing.return_value

        with patch.object(
            YtDlpDownloader, "__init__", return_value=None
        ) as downloader_constructor:
            output_dir = run_indexing(transcript_path, settings)

        create_transcriber.assert_not_called()
        downloader_constructor.assert_not_called()
        registry_constructor.load_builtin.assert_called_once_with()
        registry.resolve.assert_called_once_with(("indexacao",))
        processor_constructor.assert_called_once_with(
            api_key=settings.openai_api_key,
            default_model=settings.index_model,
        )
        runner_constructor.assert_called_once_with({"openai": processor})
        pipeline_constructor.for_processing.assert_called_once_with(
            runner=runner,
            task_registry=registry,
            output_dir=transcript_path.parent,
            progress=print,
        )
        pipeline.process_transcript_file.assert_called_once_with(
            transcript_path,
            selected_tasks,
        )
        self.assertEqual(output_dir, transcript_path.parent)


if __name__ == "__main__":
    unittest.main()
