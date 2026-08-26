import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import build_parser, run_indexing, run_transcription
from src.config import Settings


class CliTests(unittest.TestCase):
    def test_accepts_process_transcribe_and_index_commands(self) -> None:
        parser = build_parser()

        process_args = parser.parse_args(["process", "episode.mp3"])
        transcribe_args = parser.parse_args(["transcribe", "episode.mp3"])
        index_args = parser.parse_args(["index", "outputs/episode/transcript.json"])

        self.assertEqual(process_args.command, "process")
        self.assertEqual(transcribe_args.command, "transcribe")
        self.assertEqual(index_args.command, "index")
        self.assertEqual(index_args.transcript_path.name, "transcript.json")

    @patch("src.cli.create_transcriber")
    def test_transcribe_command_uses_central_provider_factory(self, factory) -> None:
        transcriber = factory.return_value
        settings = Settings(transcription_provider="local")

        output_dir = run_transcription(
            Path("episode.mp3"), Path("outputs"), settings
        )

        factory.assert_called_once_with(
            settings=settings,
            output_dir=Path("outputs/episode"),
            progress=print,
        )
        transcriber.transcribe.assert_called_once_with(Path("episode.mp3"))
        self.assertEqual(output_dir, Path("outputs/episode"))

    @patch("src.cli.PodcastPipeline")
    @patch("src.cli.OpenAIIndexer")
    @patch("src.cli.create_transcriber")
    def test_reindex_command_does_not_create_transcriber(
        self,
        create_transcriber,
        indexer_constructor,
        pipeline_constructor,
    ) -> None:
        transcript_path = Path("outputs/episode/transcript.json")
        settings = Settings()
        pipeline = pipeline_constructor.return_value

        output_dir = run_indexing(transcript_path, settings)

        create_transcriber.assert_not_called()
        indexer_constructor.assert_called_once_with(
            api_key=settings.openai_api_key,
            model=settings.index_model,
        )
        pipeline.index_transcript_file.assert_called_once_with(transcript_path)
        self.assertEqual(output_dir, transcript_path.parent)


if __name__ == "__main__":
    unittest.main()
