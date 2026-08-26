# podcast-indexer

MVP local em Python para transcrever um arquivo de áudio com timestamps e gerar
uma decupagem cronológica do episódio.

## Arquitetura

- `src/models.py`: modelos internos independentes de provider (`Transcript` e
  `IndexResult`).
- `src/transcription/`: contrato `Transcriber` e adaptador da OpenAI.
- `src/indexing/`: contrato `Indexer` e adaptador da OpenAI usando Responses API
  com Structured Outputs.
- `src/prompting.py`: carregamento e substituição previsível de
  `{{TRANSCRIPT}}`.
- `src/prompts/decupagem.md`: prompt editável de decupagem.
- `src/pipeline.py`: orquestração neutra em relação aos providers e persistência
  local.
- `src/cli.py`: comandos de transcrição, processamento completo e reindexação.

Os providers dependem apenas dos contratos `Transcriber` e `Indexer`. Assim, eles
podem ser substituídos sem levar tipos do SDK da OpenAI para o restante da
aplicação.

## Preparação

Requer Python 3.11 ou mais recente.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Configure `OPENAI_API_KEY` no `.env`. Os modelos são centralizados pelas
variáveis `TRANSCRIPTION_MODEL` e `INDEX_MODEL`; o indexador usa
`gpt-5-mini` por padrão. O `.env` é ignorado pelo Git.

## Uso

Pipeline completo:

```powershell
python -m src.cli process "D:\Podcasts\episodio.mp3"
```

Somente transcrição:

```powershell
python -m src.cli transcribe "D:\Podcasts\episodio.mp3"
```

Reindexação sem nova chamada de transcrição:

```powershell
python -m src.cli index "outputs\episodio\transcript.json"
```

Os comandos que recebem áudio aceitam `--output-dir`. O processamento completo
cria `outputs/<episódio>/` contendo:

- `transcript.raw.json`: resposta serializável do provider de transcrição;
- `transcript.json`: transcrição interna normalizada;
- `transcript.txt`: falas com timestamps;
- `index.json`: `IndexResult` normalizado;
- `index.md`: visualização legível dos assuntos.

O comando `index` grava `index.json` e `index.md` ao lado do `transcript.json`, o
que permite editar o prompt e reindexar sem pagar novamente pela transcrição.

## Testes

Todos os testes usam providers simulados ou mocks e não fazem chamadas pagas:

```powershell
python -m unittest discover -s tests -v
```

Com as dependências de desenvolvimento instaladas, também é possível usar:

```powershell
python -m pytest
```

## Diagnóstico do faster-whisper (CUDA)

O teste local é isolado do pipeline principal e exige uma GPU NVIDIA com as
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
