# Enterprise AI Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) application for searching, summarizing and comparing enterprise data-center, infrastructure and sustainability reports.

## Overview

The application indexes PDF reports in a local ChromaDB vector database and answers user questions using retrieved document excerpts. Documents, embeddings and model inference remain local.

## Architecture

### Document ingestion

```text
PDF Reports
    ↓
Read File
    ↓
Split Text
    ↓
nomic-embed-text
    ↓
ChromaDB
```

### Question answering

```text
User Question
    ↓
Streamlit
    ↓
Langflow REST API
    ↓
Question Embedding
    ↓
ChromaDB Retrieval
    ↓
Retrieved Context
    ↓
Prompt Template
    ↓
Qwen 2.5 through Ollama
    ↓
Generated Answer
```

## Technology stack

- Python
- Streamlit
- Langflow
- Ollama
- Qwen 2.5 3B
- nomic-embed-text
- ChromaDB
- Requests
- python-dotenv

## Features

- Local RAG pipeline
- Multi-document PDF ingestion
- Text chunking with overlap
- Local embeddings
- Persistent vector storage
- Semantic/MMR retrieval
- Grounded prompt orchestration
- Local LLM inference
- Browser chat interface
- Conversation history and export
- Suggested prompts
- AI workflow visualization
- Project evidence tab
- Error and timeout handling

## Screenshots

Add these files to the `screenshots` folder:

- `01-document-ingestion-flow.png`
- `02-question-answering-flow.png`
- `03-enterprise-ai-interface.png`
- `04-working-answer.png`

## Run locally

### 1. Start Ollama

```powershell
ollama serve
```

### 2. Start Langflow

```powershell
langflow run
```

### 3. Start Streamlit

```powershell
.\.venv\Scripts\Activate
streamlit run app.py
```

Open `http://localhost:8501`.

## Environment variables

Copy `.env.example` to `.env` and add your actual credentials.

```env
LANGFLOW_URL=http://localhost:7860
LANGFLOW_FLOW_ID=YOUR_FLOW_ID
LANGFLOW_API_KEY=YOUR_API_KEY
```

Never commit `.env` or API keys publicly.

## Portfolio summary

Designed and implemented a local enterprise Retrieval-Augmented Generation assistant using Langflow, Ollama, ChromaDB, Qwen and Streamlit. Built independent document-ingestion and question-answering workflows, integrated semantic retrieval with grounded prompt generation, and exposed the solution through a browser-based Python application.

## Known limitations

- Source filenames and page citations are not yet displayed.
- Retrieval quality depends on document structure and query wording.
- Qwen 2.5 3B may occasionally misclassify retrieved information.
- Local inference can be slow on CPU-only systems.

## Future improvements

- Source and page citations
- PDF upload and automatic ingestion
- Streaming answers
- Authentication
- Automated RAG evaluation
- Larger local model
