# AGENTS.md

## Project purpose

This project automates long-form YouTube/podcast processing for a real production workflow.

Current high-level flow:

```text
local file / YouTube URL
        ↓
InputResolver / yt-dlp
        ↓
audio
        ↓
Transcriber
        ↓
normalized Transcript
        ↓
ProcessingRunner
        ↓
one or more independent Tasks
        ↓
task-specific outputs
```

Current tasks include `indexacao` and `shorts`.

Reliability and preservation of existing production behavior are more important than architectural novelty.

---

## Working principles

1. Inspect the relevant implementation before editing.
2. Preserve existing behavior unless the request explicitly changes it.
3. Prefer small, incremental changes over large rewrites.
4. Do not combine unrelated refactors with a feature.
5. Reuse existing abstractions before creating new ones.
6. Keep the architecture modular, explicit, and easy to debug.
7. Add complexity only when the current requirement needs it.
8. The repository is the source of truth if a prompt describes outdated architecture.

Avoid premature introduction of:

- workflow/DAG engines;
- LangGraph or similar orchestration frameworks;
- autonomous-agent frameworks;
- plugin systems with arbitrary dynamic imports;
- dependency-injection frameworks;
- reflection-heavy architecture;
- databases or queues;
- distributed execution;
- async/parallel execution without a concrete requirement;
- complex caching.

The current processing model is intentionally:

```text
1 Transcript -> N independent tasks
```

Execution is sequential unless explicitly changed.

---

## Architecture boundaries

Keep these concerns separate:

1. input acquisition/resolution;
2. transcription;
3. normalized transcript representation;
4. transcript processing;
5. output persistence;
6. CLI composition.

Do not move task-specific behavior into unrelated layers.

### Input

Inputs may be local file paths or HTTP/HTTPS URLs.

URL inputs are resolved through yt-dlp.

Rules:

- never delete a user-provided local file;
- temporary downloads may be cleaned after use;
- preserve existing yt-dlp authentication/cookie behavior;
- do not change downloader behavior as part of unrelated features.

### Transcription

All transcription providers must produce the same provider-independent `Transcript`.

Downstream code must not depend on provider-specific raw transcription responses.

Production local transcription currently uses faster-whisper with the established CUDA configuration.

Do not change Whisper model, CUDA behavior, compute type, VAD, batching, decoding parameters, or fallback policy as part of unrelated work.

Do not silently fall back from CUDA to CPU.

### Transcript

`Transcript` is the stable boundary between transcription and downstream processing.

Existing stored `transcript.json` files are reusable assets and may be processed again later.

Use:

```python
Transcript.as_timestamped_text()
```

as the canonical LLM representation.

Do not create another transcript formatter unless there is a demonstrated need.

Do not casually change `Transcript` / `TranscriptSegment` schemas. Any migration must consider previously stored transcripts.

---

## Task processing architecture

Downstream analyses are Tasks.

Examples:

- `indexacao`
- `shorts`

A Task describes **what** should be done.

A processor/executor describes **how** it is executed.

Important concepts:

- `TaskDefinition`
- `TaskRegistry`
- `ProcessingRunner`
- `ProcessingResult`
- processors/executors
- explicit output schemas
- `ProcessingOutputWriter`

### Rules

The runner must not know semantic details about individual tasks.

Do not add patterns such as:

```python
if task_id == "indexacao":
    ...
elif task_id == "shorts":
    ...
```

to the main pipeline for normal task registration.

The runner must not directly depend on OpenAI.

Tasks must not mutate the shared Transcript.

All selected tasks receive the same logical Transcript.

### Adding a normal task

A normal prompt-based task should usually require only:

```text
src/tasks/<task_id>/
├── __init__.py
├── task.yaml
└── prompt.md
```

Reuse an existing result schema when it accurately represents the task.

If a genuinely new result structure is required:

1. create the model;
2. add it to the explicit schema registry;
3. add a custom Markdown renderer only if generic rendering is insufficient.

Adding a normal task should not require changes to `pipeline.py` or `cli.py`.

### Schema resolution

Prefer explicit mappings such as:

```text
index_result  -> IndexResult
shorts_result -> ShortsResult
```

Do not replace this with arbitrary imports, reflection, or a complex plugin framework without an explicit requirement.

---

## Prompts and task resources

Task definitions and prompts live under `src/tasks/`.

Prompts must remain external to Python code.

The generic OpenAI processor must not contain:

- hardcoded indexing instructions;
- hardcoded Shorts instructions;
- task-specific Markdown/JSON writing logic.

Task resources are loaded as package resources.

When adding or moving `.md` / `.yaml` task resources:

- preserve package-data configuration;
- verify resources are available from a built/installed package when relevant;
- do not replace package-resource loading with repository-specific absolute paths.

---

## Outputs and compatibility

Task outputs are independent.

Current structure is approximately:

```text
outputs/<episode>/
├── transcript.json
├── transcript.txt
├── index.json
├── index.md
└── processing/
    ├── indexacao/
    │   ├── result.json
    │   └── result.md
    └── shorts/
        ├── result.json
        └── result.md
```

`index.json` and `index.md` are compatibility outputs for the historical indexing workflow.

Do not remove or change compatibility outputs unless explicitly requested.

`result.json` should contain the task payload rather than an unnecessary nested duplicate of the complete `ProcessingResult`.

Output persistence belongs in the output layer, not inside LLM executors.

---

## Reprocessing guarantees

Processing an existing `transcript.json` must not initialize unnecessary acquisition/transcription components.

A transcript-only processing flow must not require:

- yt-dlp;
- remote input resolution;
- faster-whisper;
- a transcription API.

This is an important production feature.

---

## Failure behavior

Processing is currently:

- sequential;
- ordered;
- fail-fast.

If one task fails:

- clearly identify the failed task;
- preserve the original exception as the cause when appropriate;
- do not silently continue to subsequent tasks;
- preserve outputs already written by successful previous tasks.

Do not change this policy incidentally.

Errors must not expose:

- API keys;
- cookie contents;
- authentication secrets.

---

## Backward compatibility

This application is used operationally.

Before changing any existing:

- CLI command or argument;
- constructor;
- return type;
- filename;
- JSON schema;
- Markdown format;
- cleanup behavior;

determine whether it is already relied upon.

Prefer adapters or compatibility aliases for inexpensive migrations.

Do not remove legacy adapters merely because they appear redundant. Remove compatibility code only in an explicit cleanup task after confirming there are no remaining consumers.

---

## CLI responsibilities

The CLI may:

- parse input;
- resolve task ids;
- compose application components;
- display concise progress and errors.

It should not contain task-specific business logic.

Validate requested task ids before constructing expensive components whenever practical.

Default CLI behavior must remain compatible with the established workflow unless the request explicitly changes it.

---

## Testing policy

Meaningful changes require tests.

### Before a non-trivial change

1. run relevant existing tests;
2. record the baseline;
3. distinguish pre-existing failures from regressions;
4. add characterization tests first if existing behavior is poorly covered.

### After a change

1. run focused tests;
2. run the full test suite for non-trivial changes;
3. report the actual result.

Do not:

- delete tests to hide regressions;
- weaken assertions merely to make tests pass;
- mark new failures as expected without justification;
- claim a validation was run when it was not.

Unit tests must not make real OpenAI requests.

Unit tests must not make real YouTube downloads.

Use mocks/fakes/stubs at external boundaries.

Changes involving task package resources should validate resource loading from a built/installed package when appropriate.

---

## Dependency policy

Prefer the standard library and dependencies already present.

When adding a dependency:

- explain why it is needed;
- keep it narrowly scoped;
- register it correctly in `pyproject.toml`;
- avoid large frameworks for small problems.

Do not replace working libraries simply because another library is newer or more fashionable.

---

## Code quality

Follow the conventions already present in the repository.

Prefer:

- descriptive names;
- explicit data flow;
- type hints consistent with the codebase;
- small units with one clear responsibility;
- Pydantic where the project already uses validated Pydantic models;
- simple control flow.

Avoid:

- speculative abstractions;
- unnecessary base classes;
- hidden global state;
- duplicated business rules;
- duplicated transcript/prompt formatting;
- task-specific conditionals scattered through infrastructure.

Comments should explain non-obvious constraints or decisions, not restate obvious code.

---

## Safe change workflow

For significant work:

1. inspect the relevant code and tests;
2. identify current contracts and invariants;
3. propose the smallest safe change;
4. add/confirm regression coverage;
5. implement only the requested scope;
6. run focused tests;
7. run the full suite;
8. summarize compatibility impact and limitations.

Normally keep these as separate changes:

- adding a new processing task;
- changing faster-whisper configuration;
- changing yt-dlp behavior;
- removing compatibility adapters;
- reorganizing output paths;
- changing CLI syntax;
- introducing concurrency.

When asked only for analysis or a plan, do not modify files.

When asked to implement one phase, do not continue into later phases without explicit instruction.

---

## Definition of done

Before reporting a coding task complete:

- confirm the requested behavior exists;
- run the appropriate tests;
- report test results;
- report pre-existing failures separately;
- confirm unrelated protected components were not modified;
- summarize files created/changed;
- describe compatibility impact;
- describe deliberate limitations.

Never claim a test, build, API call, download, or runtime validation was performed unless it was actually executed.
