import unittest

from pydantic import ValidationError

from src.tasks.shorts.models import ShortMoment, ShortsResult


class ShortsModelTests(unittest.TestCase):
    def test_builds_task_specific_result_and_normalizes_optional_text(self) -> None:
        result = ShortsResult(
            moments=[
                ShortMoment(
                    start_seconds=10.25,
                    end_seconds=24.75,
                    title="  Momento de teste  ",
                    summary="  Resumo  ",
                    central_idea=None,
                    context="   ",
                )
            ]
        )

        self.assertEqual(result.moments[0].title, "Momento de teste")
        self.assertEqual(result.moments[0].summary, "Resumo")
        self.assertIsNone(result.moments[0].central_idea)
        self.assertIsNone(result.moments[0].context)

    def test_allows_an_empty_moment_list(self) -> None:
        self.assertEqual(ShortsResult().moments, [])

    def test_rejects_reversed_or_non_finite_ranges(self) -> None:
        with self.assertRaises(ValidationError):
            ShortMoment(start_seconds=20, end_seconds=10)
        with self.assertRaises(ValidationError):
            ShortMoment(start_seconds=0, end_seconds=float("inf"))


if __name__ == "__main__":
    unittest.main()
