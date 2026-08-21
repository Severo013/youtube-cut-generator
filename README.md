# podcast-indexer

MVP local em Python para transcrever um arquivo de áudio, normalizar a
transcrição com timestamps, enviá-la a um indexador e salvar os resultados em
JSON. Neste primeiro passo, os adaptadores OpenAI são apenas placeholders e não
fazem chamadas de API.

## Arquitetura

- `src/models.py`: modelos internos `Transcript`, `TranscriptSegment` e
  `IndexResult`, sem tipos específicos de providers.
- `src/transcription/`: contrato `Transcriber` e futuro adaptador OpenAI.
- `src/indexing/`: contrato `Indexer` e futuro adaptador OpenAI.
- `src/prompting.py` e `src/prompts/`: carregamento, template e renderização do
  prompt de decupagem.
- `src/pipeline.py`: orquestração e persistência em `outputs/<episódio>/`.
- `src/config.py`: configuração por `.env` e variáveis de ambiente.
- `src/cli.py`: ponto de entrada da linha de comando.

As dependências apontam para contratos simples (`Protocol`), portanto um novo
provider só precisa implementar `transcribe(...)` ou `index(...)`.

## Preparação

Requer Python 3.11 ou mais recente.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Preencha `OPENAI_API_KEY` apenas no arquivo `.env`, que é ignorado pelo Git.

## Uso

```powershell
python -m src.cli "D:\Podcasts\episodio.mp3"
```

Por enquanto o comando valida a entrada e termina com uma mensagem informando
que a integração OpenAI ainda não foi implementada. Quando os adaptadores forem
conectados, os arquivos `transcript.json` e `index.json` serão gravados em
`outputs/<episódio>/`.

## Testes

Sem instalar ferramentas adicionais:

```powershell
python -m unittest discover -s tests -v
```

Ou, após instalar as dependências de desenvolvimento:

```powershell
python -m pytest
```
