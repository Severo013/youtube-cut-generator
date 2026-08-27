# podcast-indexer

MVP local em Python para transcrever um arquivo de áudio com timestamps e gerar
uma decupagem cronológica do episódio.

## Arquitetura

- `src/models.py`: modelos internos independentes de provider (`Transcript` e
  `IndexResult`).
- `src/transcription/`: contrato `Transcriber`, adapters do faster-whisper e da
  OpenAI, e factory central de seleção do provider.
- `src/indexing/`: contrato `Indexer` e adaptador da OpenAI usando Responses API
  com Structured Outputs.
- `src/prompting.py`: carregamento e substituição previsível de
  `{{TRANSCRIPT}}`.
- `src/prompts/decupagem.md`: prompt editável de decupagem.
- `src/pipeline.py`: orquestração neutra em relação aos providers e persistência
  local.
- `src/input_resolution.py`: resolve caminhos locais e baixa URLs para um
  diretório temporário exclusivo usando a API Python do `yt-dlp`.
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
continua obrigatória para `process` e `index`, pois a indexação usa
`gpt-5-mini`. O `.env` é ignorado pelo Git.

Para voltar à transcrição pela OpenAI sem alterar código:

```dotenv
TRANSCRIPTION_PROVIDER=openai
TRANSCRIPTION_MODEL=whisper-1
```

## Uso

Pipeline completo:

```powershell
python -m src.cli process "D:\Podcasts\episodio.mp3"
```

O mesmo comando aceita uma URL HTTP/HTTPS e ignora parâmetros de playlist para
baixar somente o vídeo solicitado:

```powershell
python -m src.cli process "https://www.youtube.com/watch?v=..."
```

O yt-dlp seleciona `bestaudio/best` e entrega o arquivo original baixado ao
transcriber, sem conversão para MP3, pós-processador de áudio ou FFmpeg externo.
Por padrão o diretório temporário é removido mesmo quando uma etapa posterior
falha. Use `YTDLP_KEEP_TEMP=true` apenas para depuração; o caminho preservado é
mostrado no terminal.

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

- `transcript.raw.json`: metadados e segmentos serializáveis do provider de
  transcrição, identificados por `provider` e `model`;
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
