import unittest

from src.cli import build_parser


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


if __name__ == "__main__":
    unittest.main()
