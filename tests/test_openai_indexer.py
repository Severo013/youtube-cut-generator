import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from openai import OpenAIError

from src.indexing import (
    EmptyIndexResultError,
    IndexingConfigurationError,
    IndexingProviderError,
)
from src.indexing.openai_indexer import OpenAIIndexer
from src.models import IndexResult, Transcript, TranscriptSegment
from src.processing.openai_processor import OpenAITranscriptProcessor


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

    def test_legacy_adapter_delegates_to_generic_processor(self) -> None:
        processor = Mock(spec=OpenAITranscriptProcessor)
        expected = IndexResult.model_validate(
            {
                "blocks": [
                    {
                        "start_seconds": 5,
                        "end_seconds": 18,
                        "start_timestamp": "00:00:05",
                        "end_timestamp": "00:00:18",
                        "topic": "Tema",
                        "summary": "Resumo",
                    }
                ]
            }
        )
        processor.execute_prompt.return_value = expected
        indexer = OpenAIIndexer(
            api_key=None,
            model="ignored-by-adapter-test",
            processor=processor,
        )

        result = indexer.index(make_transcript(), "prompt legado")

        self.assertIs(result, expected)
        task, prompt = processor.execute_prompt.call_args.args
        self.assertEqual(task.id, "indexacao")
        self.assertEqual(prompt, "prompt legado")

    def test_legacy_adapter_preserves_provider_error_message(self) -> None:
        client = Mock()
        provider_error = OpenAIError("service unavailable")
        client.responses.parse.side_effect = provider_error
        indexer = OpenAIIndexer(
            api_key="test-key",
            model="gpt-5-mini",
            client=client,
        )

        with self.assertRaisesRegex(
            IndexingProviderError,
            "A API de indexação da OpenAI falhou: service unavailable",
        ):
            indexer.index(make_transcript(), "prompt")


if __name__ == "__main__":
    unittest.main()
