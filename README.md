# podcast-indexer

MVP local em Python para transcrever um arquivo de áudio, normalizar a
transcrição com timestamps e, futuramente, enviá-la a um indexador. A etapa de
transcrição usa o SDK oficial da OpenAI; a indexação ainda é um placeholder e
não faz chamadas de API.

## Arquitetura

- `src/models.py`: modelos internos `Transcript`, `TranscriptSegment` e
  `IndexResult`, sem tipos específicos de providers.
- `src/transcription/`: contrato `Transcriber`, erros de domínio e adaptador
  OpenAI.
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

## Transcrição isolada

```powershell
python -m src.cli transcribe "D:\Podcasts\episodio.mp3"
```

É possível escolher outro diretório raiz com `--output-dir`. O comando cria
`outputs/<episódio>/` com:

- `transcript.raw.json`: resposta serializável do provider para debug;
- `transcript.json`: modelo interno normalizado;
- `transcript.txt`: segmentos legíveis com timestamps `HH:MM:SS`.

A chamada usa o modelo definido por `TRANSCRIPTION_MODEL`, português como
idioma preferencial, resposta `verbose_json` e timestamps por segmento. O valor
padrão é `whisper-1`. Nenhum parâmetro de chunking é enviado nessa chamada.

Segundo a documentação oficial atual, o endpoint de transcrição aceita arquivos
de até 25 MB. O aplicativo rejeita antecipadamente arquivos maiores com uma
mensagem clara. Ainda não há splitting, compressão ou chunking manual; episódios
acima desse limite precisarão de preprocessing em uma etapa futura. A indexação
com modelo de linguagem permanece não implementada.

## Testes

Sem instalar ferramentas adicionais:

```powershell
python -m unittest discover -s tests -v
```

Ou, após instalar as dependências de desenvolvimento:

```powershell
python -m pytest
```
