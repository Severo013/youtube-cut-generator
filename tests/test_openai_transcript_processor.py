import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.models import IndexResult, Transcript, TranscriptSegment
from src.processing import EmptyProcessingResultError, TaskRegistry
from src.processing.openai_processor import OpenAITranscriptProcessor


def make_transcript() -> Transcript:
    return Transcript(
        source="episode.mp3",
        duration_seconds=18,
        language="pt",
        full_text="Primeira fala.",
        segments=[
            TranscriptSegment(start_seconds=5, end_seconds=18, text="Primeira fala.")
        ],
    )


def parsed_index() -> dict[str, object]:
    return {
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


class OpenAITranscriptProcessorTests(unittest.TestCase):
    def test_loads_external_prompt_and_parses_task_schema_with_mocked_client(self) -> None:
        task = TaskRegistry.load_builtin().get("indexacao")
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(
            output_parsed=parsed_index()
        )
        processor = OpenAITranscriptProcessor(
            api_key="test-key",
            default_model="gpt-5-mini",
            client=client,
        )
        transcript = make_transcript()

        with patch.object(
            type(task),
            "load_prompt",
            autospec=True,
            return_value="PROMPT EXTERNO\n{{TRANSCRIPT}}",
        ) as load_prompt:
            result = processor.process(
                task,
                transcript,
                transcript.as_timestamped_text(),
            )

        self.assertIsInstance(result, IndexResult)
        load_prompt.assert_called_once_with(task)
        client.responses.parse.assert_called_once_with(
            model="gpt-5-mini",
            input="PROMPT EXTERNO\n[00:00:05 - 00:00:18] Primeira fala.",
            text_format=IndexResult,
        )

    def test_task_model_overrides_processor_default(self) -> None:
        task = TaskRegistry.load_builtin().get("indexacao").model_copy(
            update={"model": "task-model"}
        )
        task.bind_runtime(
            resource_directory=TaskRegistry.load_builtin()
            .get("indexacao")
            .prompt_resource.parent,
            result_type=IndexResult,
        )
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(
            output_parsed=parsed_index()
        )
        processor = OpenAITranscriptProcessor(
            api_key="test-key",
            default_model="default-model",
            client=client,
        )

        processor.execute_prompt(task, "prompt")

        self.assertEqual(
            client.responses.parse.call_args.kwargs["model"],
            "task-model",
        )

    def test_rejects_response_without_parsed_output(self) -> None:
        task = TaskRegistry.load_builtin().get("indexacao")
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(output_parsed=None)
        processor = OpenAITranscriptProcessor(
            api_key="test-key",
            default_model="gpt-5-mini",
            client=client,
        )

        with self.assertRaises(EmptyProcessingResultError):
            processor.execute_prompt(task, "prompt")


if __name__ == "__main__":
    unittest.main()
