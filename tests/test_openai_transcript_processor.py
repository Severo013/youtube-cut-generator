import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.models import IndexResult, Transcript, TranscriptSegment
from src.processing import (
    EmptyProcessingResultError,
    ProcessingConfigurationError,
    ProcessingRunner,
    TaskDefinition,
    TaskRegistry,
)
from src.processing.openai_processor import OpenAITranscriptProcessor
from src.tasks.cortes.models import CutsResult
from src.tasks.shorts.models import ShortsResult


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

    def test_uses_shorts_schema_for_shorts_task(self) -> None:
        task = TaskRegistry.load_builtin().get("shorts")
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(
            output_parsed={
                "moments": [
                    {
                        "start_seconds": 5,
                        "end_seconds": 18,
                        "title": "Momento",
                        "summary": None,
                        "central_idea": None,
                        "context": None,
                    }
                ]
            }
        )
        processor = OpenAITranscriptProcessor(
            api_key="test-key",
            default_model="gpt-5-mini",
            client=client,
        )

        result = processor.execute_prompt(task, "prompt shorts")

        self.assertIsInstance(result, ShortsResult)
        client.responses.parse.assert_called_once_with(
            model="gpt-5-mini",
            input="prompt shorts",
            text_format=ShortsResult,
        )

    def test_runner_passes_an_already_rendered_prompt_to_openai_executor(self) -> None:
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
            return_value="PREPARADO\n{{TRANSCRIPT}}",
        ):
            ProcessingRunner(executors={"openai": processor}).run(
                transcript,
                [task],
            )

        self.assertEqual(
            client.responses.parse.call_args.kwargs["input"],
            "PREPARADO\n[00:00:05 - 00:00:18] Primeira fala.",
        )

    def test_legacy_process_explicitly_rejects_dependent_task(self) -> None:
        task = TaskDefinition.model_validate(
            {
                "id": "dependent",
                "name": "Dependent",
                "prompt": "prompt.md",
                "executor": "openai",
                "schema": "index_result",
                "requires": ["indexacao"],
            }
        )
        processor = OpenAITranscriptProcessor(
            api_key="test-key",
            default_model="gpt-5-mini",
            client=Mock(),
        )
        transcript = make_transcript()

        with self.assertRaisesRegex(
            ProcessingConfigurationError,
            "ProcessingRunner.*context",
        ):
            processor.process(task, transcript, transcript.as_timestamped_text())

    def test_executes_cortes_through_the_same_structured_output_contract(self) -> None:
        task = TaskRegistry.load_builtin().get("cortes")
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(
            output_parsed={
                "cuts": [
                    {
                        "start_seconds": 0,
                        "end_seconds": 600,
                        "theme": "Tema",
                        "summary": "Resumo",
                        "title": "TÍTULO",
                        "thumbnail_text": "FRASE CURTA",
                    }
                ],
                "limitations": ["Uma opção válida."],
            }
        )
        processor = OpenAITranscriptProcessor(
            api_key="test-key",
            default_model="gpt-5-mini",
            client=client,
        )

        result = processor.execute_prompt(task, "prompt já renderizado")

        self.assertIsInstance(result, CutsResult)
        client.responses.parse.assert_called_once_with(
            model="gpt-5-mini",
            input="prompt já renderizado",
            text_format=CutsResult,
        )


if __name__ == "__main__":
    unittest.main()
