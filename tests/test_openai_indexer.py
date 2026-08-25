import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.indexing import EmptyIndexResultError, IndexingConfigurationError
from src.indexing.openai_indexer import OpenAIIndexer
from src.models import IndexResult, Transcript, TranscriptSegment


def make_transcript() -> Transcript:
    return Transcript(
        source="episode.mp3",
        duration_seconds=18,
        language="pt",
        full_text="Primeira fala. Segunda fala.",
        segments=[
            TranscriptSegment(start_seconds=5, end_seconds=11, text="Primeira fala"),
            TranscriptSegment(start_seconds=11, end_seconds=18, text="Segunda fala"),
        ],
    )


class OpenAIIndexerTests(unittest.TestCase):
    def test_parses_structured_output_into_internal_index_result(self) -> None:
        parsed_payload = {
            "blocks": [
                {
                    "start_seconds": 5,
                    "end_seconds": 18,
                    "start_timestamp": "00:00:05",
                    "end_timestamp": "00:00:18",
                    "topic": "Apresentação",
                    "summary": "Os participantes se apresentam.",
                }
            ]
        }
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(
            output_parsed=parsed_payload
        )
        indexer = OpenAIIndexer(
            api_key="test-key",
            model="gpt-5-mini",
            client=client,
        )

        result = indexer.index(make_transcript(), "prompt renderizado")

        self.assertIsInstance(result, IndexResult)
        self.assertEqual(result.blocks[0].topic, "Apresentação")
        client.responses.parse.assert_called_once_with(
            model="gpt-5-mini",
            input="prompt renderizado",
            text_format=IndexResult,
        )

    def test_rejects_response_without_parsed_output(self) -> None:
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(output_parsed=None)
        indexer = OpenAIIndexer(
            api_key="test-key",
            model="gpt-5-mini",
            client=client,
        )

        with self.assertRaises(EmptyIndexResultError):
            indexer.index(make_transcript(), "prompt")

    def test_rejects_missing_api_key_before_calling_api(self) -> None:
        client = Mock()
        indexer = OpenAIIndexer(
            api_key=None,
            model="gpt-5-mini",
            client=client,
        )

        with self.assertRaises(IndexingConfigurationError):
            indexer.index(make_transcript(), "prompt")

        client.responses.parse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
