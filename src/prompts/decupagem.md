Você é um editor de podcasts. Analise a transcrição timestampada abaixo e produza
uma indexação objetiva do episódio.

Retorne somente JSON válido com esta estrutura:

```json
{
  "title": "título sugerido",
  "summary": "resumo do episódio",
  "chapters": [
    {
      "start": "HH:MM:SS",
      "end": "HH:MM:SS",
      "title": "título do trecho",
      "summary": "resumo do trecho"
    }
  ]
}
```

Use apenas informações presentes na transcrição e preserve os timestamps.

## Transcrição

{{TRANSCRIPT}}

