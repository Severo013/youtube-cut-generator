# podcast-indexer

MVP local em Python para transcrever um arquivo de áudio com timestamps e gerar
uma decupagem cronológica do episódio.

## Arquitetura

- `src/models.py`: modelos internos independentes de provider (`Transcript` e
  `IndexResult`).
- `src/transcription/`: contrato `Transcriber`, adapters do faster-whisper e da
  OpenAI, e factory central de seleção do provider.
- `src/processing/`: definições e registry de tasks, runner sequencial e
  processor genérico da OpenAI usando Responses API com Structured Outputs.
- `src/tasks/`: definições externas e prompts das tasks `indexacao` e `shorts`.
- `src/indexing/`: contratos e adapter temporário que preservam a API antiga do
  `Indexer` sobre a camada genérica.
- `src/prompting.py`: carregamento e substituição previsível de
  `{{TRANSCRIPT}}`.
- `src/pipeline.py`: orquestração da transcrição e execução sequencial das tasks
  selecionadas.
- `src/input_resolution.py`: resolve caminhos locais e baixa URLs para um
  diretório temporário exclusivo usando a API Python do `yt-dlp`.
- `src/cli.py`: comandos de transcrição, processamento completo e reindexação.

Cada `TaskDefinition` declara id, nome, prompt, executor e schema de saída. O
`ProcessingRunner` usa o executor cadastrado sem conhecer OpenAI ou a semântica
das tasks. A execução é sequencial e fail-fast. O contrato antigo `Indexer`
permanece disponível como adapter de compatibilidade.

## Preparação

Requer Python 3.11 ou mais recente.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

O extra oficial `yt-dlp[default]` é instalado com o projeto e inclui o pacote
`yt-dlp-ejs`. Para suporte completo ao YouTube, instale também o Deno 2.3 ou
mais recente e confirme que `deno.exe` está no `PATH`. O Deno é o runtime
JavaScript recomendado e habilitado por padrão pelo yt-dlp.

O provider de transcrição padrão é o faster-whisper local. A configuração
recomendada no `.env` é:

```dotenv
TRANSCRIPTION_PROVIDER=local
LOCAL_WHISPER_MODEL=turbo
LOCAL_WHISPER_DEVICE=cuda
LOCAL_WHISPER_COMPUTE_TYPE=float16
TRANSCRIPTION_LANGUAGE=pt

OPENAI_API_KEY=
TRANSCRIPTION_MODEL=whisper-1
INDEX_MODEL=gpt-5-mini

YTDLP_KEEP_TEMP=false
YTDLP_AUTH_MODE=auto
YTDLP_BROWSER=firefox
YTDLP_COOKIES_FILE=
```

Os modos de autenticação de URL são `none`, `browser`, `file` e `auto`. Em
`auto`, o download é tentado primeiro sem cookies e, somente se falhar, ocorre
uma segunda tentativa. `YTDLP_COOKIES_FILE`, quando definido, tem precedência
sobre `YTDLP_BROWSER` nesse fallback. Arquivos de cookies devem estar no
formato Netscape aceito pelo yt-dlp e nunca são copiados para `outputs/`.

`OPENAI_API_KEY` não é necessária para `transcribe` com o provider local. Ela
continua obrigatória para tasks executadas pela OpenAI em `process`,
`process-transcript` e `index`. O modelo padrão continua sendo `INDEX_MODEL`. O
`.env` é ignorado pelo Git.

Para voltar à transcrição pela OpenAI sem alterar código:

```dotenv
TRANSCRIPTION_PROVIDER=openai
TRANSCRIPTION_MODEL=whisper-1
```

## Uso

Processar somente indexação:

```powershell
python -m src.cli process "<url>" --tasks indexacao
```

Indexação e seleção de Shorts sobre a mesma transcrição:

```powershell
python -m src.cli process "<url>" --tasks indexacao shorts
```

Somente Shorts:

```powershell
python -m src.cli process "<url>" --tasks shorts
```

Arquivo local:

```powershell
python -m src.cli process "D:\arquivo.mp3" --tasks indexacao shorts
```

Sem `--tasks`, `process` continua executando somente `indexacao`.

O comando aceita URLs HTTP/HTTPS e ignora parâmetros de playlist para baixar
somente o vídeo solicitado.

O yt-dlp seleciona `bestaudio/best` e entrega o arquivo original baixado ao
transcriber, sem conversão para MP3, pós-processador de áudio ou FFmpeg externo.
Por padrão o diretório temporário é removido mesmo quando uma etapa posterior
falha. Use `YTDLP_KEEP_TEMP=true` apenas para depuração; o caminho preservado é
mostrado no terminal.

Somente transcrição:

```powershell
python -m src.cli transcribe "D:\Podcasts\episodio.mp3"
```

Executar novas tasks sobre uma transcrição existente, sem downloader ou Whisper:

```powershell
python -m src.cli process-transcript "outputs\episodio\transcript.json" --tasks shorts
```

Também é possível executar várias tasks:

```powershell
python -m src.cli process-transcript "outputs\episodio\transcript.json" --tasks indexacao shorts
```

O comando legado de reindexação continua disponível:

```powershell
python -m src.cli index "outputs\episodio\transcript.json"
```

Listar as tasks descobertas pelo registry:

```powershell
python -m src.cli list-tasks
```

Os comandos que recebem áudio aceitam `--output-dir`. Quando a entrada é uma
URL, `<episódio>` é o título do vídeo (com caracteres inválidos para nomes de
pasta substituídos). Para arquivos locais, continua sendo o nome do arquivo sem
a extensão.

O processamento completo cria `outputs/<episódio>/` contendo:

- `transcript.raw.json`: metadados e segmentos serializáveis do provider de
  transcrição, identificados por `provider` e `model`;
- `transcript.json`: transcrição interna normalizada;
- `transcript.txt`: falas com timestamps;
- `index.json`: `IndexResult` normalizado;
- `index.md`: visualização legível dos assuntos, preservada por compatibilidade;
- `processing/indexacao/result.json` e `result.md`;
- `processing/shorts/result.json` e `result.md`, quando `shorts` for selecionada.

O comando `index` grava `index.json` e `index.md` ao lado do `transcript.json`, o
que permite editar o prompt e reindexar sem pagar novamente pela transcrição.

Cada `result.json` contém somente o resultado específico da task. Metadata comum
como task, executor, modelo e status permanece no `ProcessingResult` em memória,
sem duplicar o payload nos arquivos.

## Criar uma nova task

Para uma task que reutilize um schema já registrado:

1. crie `src/tasks/<id>/__init__.py`;
2. crie `src/tasks/<id>/task.yaml` com `id`, `name`, `prompt`, `executor` e
   `schema`;
3. crie `src/tasks/<id>/prompt.md` contendo `{{TRANSCRIPT}}`.

Não é necessário alterar a CLI nem o pipeline. Quando a task precisar de um novo
Structured Output, crie seu modelo Pydantic e registre explicitamente o novo id
de schema em `DEFAULT_TASK_SCHEMAS`. Um renderer Markdown específico é opcional;
sem ele, o writer usa uma representação JSON genérica.

## Testes

Todos os testes usam providers simulados ou mocks e não fazem chamadas pagas:

```powershell
python -m unittest discover -s tests -v
```

Com as dependências de desenvolvimento instaladas, também é possível usar:

```powershell
python -m pytest
```

Os testes de URL usam fakes da API Python do yt-dlp e não acessam o YouTube.

## Diagnóstico do yt-dlp

Para verificar versões, EJS e a disponibilidade do Deno sem fazer download:

```powershell
python scripts/diagnose_ytdlp.py
```

Quando houver uma URL de teste, o mesmo script registra autenticação utilizada,
formato, tamanho, tempo, caminho temporário, executa a transcrição e confirma a
remoção ao final:

```powershell
python scripts/diagnose_ytdlp.py "https://www.youtube.com/watch?v=..."
```

## Diagnóstico do faster-whisper (CUDA)

O teste diagnóstico é isolado do pipeline principal e exige uma GPU NVIDIA com as
bibliotecas CUDA compatíveis disponíveis no `PATH`. Ele usa `turbo`, CUDA,
FP16, português e timestamps por segmento, sem fallback para CPU:

```powershell
python scripts/test_local_whisper.py "D:\Podcasts\amostra.mp3"
```

Para Blackwell no Windows, use CTranslate2 4.6.3 ou mais recente e o runtime
CUDA 12.8 ou mais recente dentro da série CUDA 12. O diretório `bin` do CUDA
deve estar no `PATH` e disponibilizar `cublas64_12.dll` e
`cublasLt64_12.dll`. A wheel atual do CTranslate2 inclui `cudnn64_9.dll`; o
Windows também precisa do Microsoft Visual C++ Redistributable atual.

Por padrão, o modelo usa o cache normal do Hugging Face. Para escolher outro
cache sem gravar modelos nas pastas de outputs, use `--cache-dir` ou a variável
`FASTER_WHISPER_CACHE_DIR`:

```powershell
$env:FASTER_WHISPER_CACHE_DIR = "D:\Caches\faster-whisper"
python scripts/test_local_whisper.py "D:\Podcasts\amostra.mp3"
```
