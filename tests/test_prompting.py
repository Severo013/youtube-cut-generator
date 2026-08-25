import tempfile
import unittest
from pathlib import Path

from src.models import Transcript, TranscriptSegment
from src.prompting import load_prompt, render_decupagem_prompt


class PromptingTests(unittest.TestCase):
    def test_loads_prompt_from_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            prompt_path = Path(temporary_dir) / "decupagem.md"
            prompt_path.write_text("Instruções\n{{TRANSCRIPT}}", encoding="utf-8")

            self.assertEqual(
                load_prompt(prompt_path),
                "Instruções\n{{TRANSCRIPT}}",
            )

    def test_renders_all_segments_with_timestamps(self) -> None:
        transcript = Transcript(
            source="episode.mp3",
            duration_seconds=18,
            language="pt",
            full_text="Primeira fala. Segunda fala.",
            segments=[
                TranscriptSegment(start_seconds=5, end_seconds=11, text="Primeira fala"),
                TranscriptSegment(start_seconds=11, end_seconds=18, text="Segunda fala"),
            ],
        )

        rendered = render_decupagem_prompt("Antes\n{{TRANSCRIPT}}\nDepois", transcript)

        self.assertEqual(
            rendered,
            (
                "Antes\n"
                "[00:00:05 - 00:00:11] Primeira fala\n"
                "[00:00:11 - 00:00:18] Segunda fala\n"
                "Depois"
            ),
        )

    def test_rejects_template_without_transcript_placeholder(self) -> None:
        transcript = Transcript(
            source="episode.mp3",
            duration_seconds=1,
            language="pt",
            full_text="Teste",
            segments=[TranscriptSegment(start_seconds=0, end_seconds=1, text="Teste")],
        )

        with self.assertRaisesRegex(ValueError, "TRANSCRIPT"):
            render_decupagem_prompt("Sem variável", transcript)


if __name__ == "__main__":
    unittest.main()
