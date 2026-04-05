# 🧪 Laboratory Intelligence for Normative Assistance (LINA) Chat

**Desafio SBPC/ML 2026–2027** · RAG + LLM para Medicina Laboratorial

Ferramenta de recuperação e resposta baseada em documentos laboratoriais (POPs, validações, manuais, diretrizes). Utiliza retrieval híbrido BM25 + embeddings semânticos com geração via LLM gratuito.

## Funcionalidades

- Upload de PDF, RMD, MD, TXT
- Retrieval híbrido: BM25 (léxico) + FAISS (semântico) com fusão RRF
- Rastreabilidade ISO 15189: versão, data e número de página em cada citação
- Detecção de conflitos entre documentos
- Score de confiança por resposta
- Interface pastel amigável via Streamlit

## Como usar

1. Acesse o app em: `https://labdocs-brasil.streamlit.app`
2. Cole sua API key do [OpenRouter](https://openrouter.ai) (gratuito, sem cartão)
3. Faça upload dos seus documentos
4. Clique em **Indexar documentos**
5. Faça sua pergunta

## Stack

- `sentence-transformers` — embeddings multilíngues (paraphrase-multilingual-mpnet-base-v2)
- `faiss-cpu` — busca por similaridade vetorial
- `rank-bm25` — busca léxica
- `streamlit` — interface web
- OpenRouter API — LLM gratuito (Qwen, LLaMA, etc.)
