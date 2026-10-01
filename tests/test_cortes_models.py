import unittest

from pydantic import ValidationError

from src.models import IndexBlock, IndexResult, Transcript
from src.processing import ProcessingContext, ProcessingResult, TaskRegistry
from src.tasks.cortes.models import CutSelection, CutsResult


def make_index(*ranges: tuple[int, int]) -> IndexResult:
    return IndexResult.model_validate(
        {
            "blocks": [
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "start_timestamp": f"00:{start // 60:02d}:{start % 60:02d}",
                    "end_timestamp": f"00:{end // 60:02d}:{end % 60:02d}",
                    "topic": f"Tema {position}",
                    "summary": f"Resumo {position}",
                }
                for position, (start, end) in enumerate(ranges, start=1)
            ]
        }
    )


def make_cut(start: int = 0, end: int = 600) -> dict[str, object]:
    return {
        "start_seconds": start,
        "end_seconds": end,
        "theme": " Tema principal ",
        "summary": " Resumo do corte ",
        "title": " TÍTULO DO CORTE ",
        "thumbnail_text": " FRASE CURTA ",
    }


def validation_context(index_result: IndexResult, **settings):
    registry = TaskRegistry.load_builtin()
    task = registry.get("cortes")
    if settings:
        task = task.model_copy(update={"settings": {**task.settings, **settings}})
    transcript = Transcript(
        source="episode.mp3",
        duration_seconds=1800,
        language="pt",
        full_text="",
        segments=[],
    )
    context = ProcessingContext(transcript)
    context.record(
        ProcessingResult(
            task_id="indexacao",
            task_name="Indexação do episódio",
            executor="fake",
            schema_id="index_result",
            output=index_result,
        )
    )
    return {"task_definition": task, "processing_context": context}


class CutsModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = make_index((0, 300), (300, 600), (600, 900), (900, 1200))

    def test_builds_valid_result_and_calculates_duration_without_serializing_it(self) -> None:
        result = CutsResult.model_validate(
            {
                "cuts": [make_cut()],
                "limitations": ["Só havia uma opção editorialmente válida."],
            },
            context=validation_context(self.index),
        )

        self.assertEqual(result.cuts[0].start_seconds, 0)
        self.assertEqual(result.cuts[0].end_seconds, 600)
        self.assertEqual(result.cuts[0].duration_seconds, 600)
        self.assertEqual(result.cuts[0].theme, "Tema principal")
        self.assertNotIn("duration", result.model_dump()["cuts"][0])

    def test_requires_strict_non_negative_integer_start(self) -> None:
        for value in (-1, 1.5, "1", True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                CutSelection.model_validate(make_cut(start=value))

    def test_requires_end_after_start(self) -> None:
        for end in (0, -1):
            with self.subTest(end=end), self.assertRaises(ValidationError):
                CutSelection.model_validate(make_cut(start=0, end=end))

    def test_rejects_blank_editorial_fields_and_limitations(self) -> None:
        for field in ("theme", "summary", "title", "thumbnail_text"):
            payload = make_cut()
            payload[field] = "   "
            with self.subTest(field=field), self.assertRaises(ValidationError):
                CutSelection.model_validate(payload)

        with self.assertRaises(ValidationError):
            CutsResult.model_validate({"cuts": [], "limitations": ["  "]})

    def test_rejects_more_than_target_count(self) -> None:
        with self.assertRaisesRegex(ValidationError, "more than target_count=5"):
            CutsResult.model_validate(
                {"cuts": [make_cut()] * 6, "limitations": []},
                context=validation_context(self.index),
            )

    def test_requires_limitation_when_returning_fewer_than_target(self) -> None:
        with self.assertRaisesRegex(ValidationError, "limitations must explain"):
            CutsResult.model_validate(
                {"cuts": [make_cut()], "limitations": []},
                context=validation_context(self.index),
            )

        result = CutsResult.model_validate(
            {"cuts": [make_cut()] * 5, "limitations": []},
            context=validation_context(self.index),
        )
        self.assertEqual(len(result.cuts), 5)

    def test_accepts_inclusive_minimum_and_maximum_durations(self) -> None:
        for end in (300, 1200):
            with self.subTest(end=end):
                result = CutsResult.model_validate(
                    {
                        "cuts": [make_cut(end=end)],
                        "limitations": ["Uma opção válida."],
                    },
                    context=validation_context(self.index),
                )
                self.assertEqual(result.cuts[0].duration_seconds, end)

    def test_rejects_duration_below_minimum_and_above_maximum(self) -> None:
        for index, end in (
            (make_index((0, 299)), 299),
            (make_index((0, 1201)), 1201),
        ):
            with self.subTest(end=end), self.assertRaisesRegex(
                ValidationError,
                "duration must be between 300 and 1200 seconds",
            ):
                CutsResult.model_validate(
                    {
                        "cuts": [make_cut(end=end)],
                        "limitations": ["Uma opção válida."],
                    },
                    context=validation_context(index),
                )

    def test_requires_start_and_end_boundaries_from_index(self) -> None:
        for cut, message in (
            (make_cut(start=1, end=600), "start_seconds"),
            (make_cut(start=0, end=601), "end_seconds"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                ValidationError,
                message,
            ):
                CutsResult.model_validate(
                    {"cuts": [cut], "limitations": ["Uma opção válida."]},
                    context=validation_context(self.index),
                )

    def test_rejects_start_block_after_end_block(self) -> None:
        overlapping_index = make_index((0, 1000), (300, 1300))

        with self.assertRaisesRegex(ValidationError, "start block must not occur after"):
            CutsResult.model_validate(
                {
                    "cuts": [make_cut(start=300, end=1000)],
                    "limitations": ["Uma opção válida."],
                },
                context=validation_context(overlapping_index),
            )

    def test_validates_settings_from_task_definition(self) -> None:
        for settings, message in (
            ({"target_count": 0}, "greater than 0"),
            ({"min_duration_minutes": 0}, "greater than 0"),
            ({"max_duration_minutes": 5, "min_duration_minutes": 5}, "greater"),
        ):
            with self.subTest(settings=settings), self.assertRaisesRegex(
                ValidationError,
                message,
            ):
                CutsResult.model_validate(
                    {
                        "cuts": [make_cut()],
                        "limitations": ["Uma opção válida."],
                    },
                    context=validation_context(self.index, **settings),
                )


if __name__ == "__main__":
    unittest.main()
