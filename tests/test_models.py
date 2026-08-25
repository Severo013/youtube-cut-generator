import unittest

from pydantic import ValidationError

from src.models import (
    IndexBlock,
    IndexResult,
    Transcript,
    TranscriptSegment,
    seconds_to_timestamp,
)


class TimestampTests(unittest.TestCase):
    def test_converts_seconds_to_hh_mm_ss(self) -> None:
        cases = {
            0: "00:00:00",
            59: "00:00:59",
            60: "00:01:00",
            3661: "01:01:01",
            90061.9: "25:01:01",
        }
        for seconds, expected in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(seconds_to_timestamp(seconds), expected)

    def test_rejects_invalid_seconds(self) -> None:
        for seconds in (-1, float("inf"), float("nan")):
            with self.subTest(seconds=seconds):
                with self.assertRaises(ValueError):
                    seconds_to_timestamp(seconds)

        with self.assertRaises(TypeError):
            seconds_to_timestamp("10")  # type: ignore[arg-type]


class TranscriptModelTests(unittest.TestCase):
    def test_builds_normalized_transcript_and_timestamped_text(self) -> None:
        transcript = Transcript(
            source="episode.mp3",
            duration_seconds=15.5,
            language="pt",
            full_text="Olá mundo.",
            segments=[
                TranscriptSegment(
                    start_seconds=1.2,
                    end_seconds=4.8,
                    text="  Olá mundo.  ",
                )
            ],
        )

        self.assertEqual(transcript.segments[0].text, "Olá mundo.")
        self.assertEqual(
            transcript.as_timestamped_text(),
            "[00:00:01 - 00:00:04] Olá mundo.",
        )

    def test_rejects_reversed_segment_range(self) -> None:
        with self.assertRaises(ValidationError):
            TranscriptSegment(start_seconds=10, end_seconds=5, text="Inválido")

    def test_rejects_blank_segment_text(self) -> None:
        with self.assertRaises(ValidationError):
            TranscriptSegment(start_seconds=0, end_seconds=1, text="   ")

    def test_rejects_non_finite_segment_timestamp(self) -> None:
        with self.assertRaises(ValidationError):
            TranscriptSegment(start_seconds=0, end_seconds=float("inf"), text="Teste")


class IndexModelTests(unittest.TestCase):
    def test_rejects_timestamp_that_does_not_match_seconds(self) -> None:
        with self.assertRaises(ValidationError):
            IndexBlock(
                start_seconds=5,
                end_seconds=10,
                start_timestamp="00:00:06",
                end_timestamp="00:00:10",
                topic="Tema",
                summary="Resumo",
            )

    def test_rejects_blocks_out_of_chronological_order(self) -> None:
        with self.assertRaises(ValidationError):
            IndexResult(
                blocks=[
                    IndexBlock(
                        start_seconds=10,
                        end_seconds=12,
                        start_timestamp="00:00:10",
                        end_timestamp="00:00:12",
                        topic="Segundo",
                        summary="Resumo",
                    ),
                    IndexBlock(
                        start_seconds=0,
                        end_seconds=5,
                        start_timestamp="00:00:00",
                        end_timestamp="00:00:05",
                        topic="Primeiro",
                        summary="Resumo",
                    ),
                ]
            )


if __name__ == "__main__":
    unittest.main()
