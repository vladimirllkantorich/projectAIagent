# DevVault AI

DevVault AI is a beginner-friendly Streamlit app for storing and asking questions about documents and code. It uses ChromaDB for local RAG search and CrewAI for the multi-agent answer workflow:

```text
User question -> Supervisor Agent -> Search Agent -> CrewAI Crew -> final answer

CrewAI Crew:
Answer Agent -> Critic Agent -> Supervisor Synthesis Agent
```

The app supports two LLM modes:

- OpenAI / ChatGPT API
- Local LM Studio through its OpenAI-compatible endpoint

## Project Structure

```text
app.py
frontend/streamlit_app.py
core/
agents/
rag/
loaders/
tools/
ui/
data/
.streamlit/secrets.toml.example
.python-version
requirements.txt
README.md
.gitignore
```

## Create A Virtual Environment

DevVault AI recommends Python 3.11. CrewAI requires Python 3.10-3.13, so do not use the old Python 3.9 environment.

On Windows with the Python launcher:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `python` is on your PATH and points to Python 3.10-3.13, this also works:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists from Python 3.9, remove it and recreate it with Python 3.11 or another CrewAI-supported Python version.

## Configure Secrets

Provider settings should be stored in the real `.streamlit/secrets.toml` file. Keep real keys out of git. The `.streamlit/secrets.toml.example` file is only a template and is never used as a secrets source.

Create `.streamlit/secrets.toml` using this shape:

```toml
[openai]
api_key = ""
model = "gpt-5-mini"
embedding_model = "text-embedding-3-small"

[local]
base_url = "http://localhost:1234/v1"
model = "local-model"
embedding_model = "nomic-embed-text-v1.5"

[app]
default_provider = "local"
embedding_provider = "local"
chroma_path = "./data/chroma_db"
max_parallel_agents = 1
```

The real `.streamlit/secrets.toml` file is ignored by git.

Environment variables or `.env` can still be used as a fallback for local overrides, but values from `.streamlit/secrets.toml` take priority.

## Run The App

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The compatibility entry point below also works:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py
```

You can also use the included launchers:

```powershell
.\run_app.ps1
```

or:

```bat
run_app.bat
```

## Health Check

Run this when the app behaves strangely:

```powershell
.\.venv\Scripts\python.exe health_check.py
```

The check verifies imports, CrewAI, Streamlit secrets, provider config, and ChromaDB startup without printing your API key.
It also reports whether the active interpreter is compatible with CrewAI.

## Use Local LM Studio Mode

1. Open LM Studio.
2. Load a chat model.
3. Start the local OpenAI-compatible server.
4. Keep the server URL as `http://localhost:1234/v1`.
5. In DevVault AI, choose `Local LM Studio` in the sidebar.

The app uses `api_key = "lm-studio"` internally for local requests. If `model = "local-model"` or `model = "auto"`, DevVault AI asks LM Studio for the loaded models and uses the first non-embedding chat model.

For better RAG search in local mode, load an embedding model in LM Studio and set `[local].embedding_model`. If the local embedding API is unavailable, DevVault AI falls back to lightweight local hash embeddings and shows a warning.

## Use OpenAI Mode

1. Add your OpenAI key to `[openai].api_key` in `.streamlit/secrets.toml`.
2. Choose `OpenAI / ChatGPT API` in the sidebar.
3. Set `[openai].model` if you want a different ChatGPT model.

If the key is missing, invalid, or the OpenAI model is misconfigured, the app shows a clear setup message in the UI.

The sidebar switches the chat provider only. The RAG index uses `[app].embedding_provider`, which defaults to `local`, so documents indexed in local mode stay available after switching to ChatGPT.

## Index And Ask

1. Upload PDF, TXT, or DOCX files and click `Index uploaded files`.
2. Enter a project folder path or use `Choose project folder`, then click `Index code folder`.
3. Ask a question in the chat.
4. Check the Agent Chat / Agent Log panel to see what each agent did.

If DevVault AI cannot find relevant information in the uploaded documents or indexed code, it says so instead of inventing an answer.

## CrewAI Agent Modes

`CrewAI standard` is the default sequential CrewAI flow:

```text
Supervisor -> Task -> Search -> CrewAI Answer Agent -> CrewAI Critic Agent -> CrewAI Supervisor Synthesis Agent
```

`CrewAI debate` uses the same sequential CrewAI process, but the Critic Agent is instructed to challenge the draft more aggressively before final synthesis. This is safer for LM Studio than true parallel or hierarchical multi-agent calls.

The RAG search step stays deterministic outside CrewAI. This keeps local LM Studio runs faster and more reliable: ChromaDB retrieves the source chunks first, then CrewAI agents reason over that retrieved context.

For short factual document questions, DevVault AI uses a faster route:

```text
Search Agent -> Document Agent -> Summary Agent
```

The Document Agent returns structured context status (`found_in_context`, `coverage`, `answer`, `evidence`, `sources`). The Summary Agent acts as the final judge and resolves contradictions before the answer is shown.

## Vector Database Tools

- `Clear database` removes DevVault AI collections and resets index history.
- `Repair database` recreates the local ChromaDB folder if the database is corrupted or incompatible.
- `Index history` in the sidebar shows recent indexing runs, source type, source name, chunk count, and status.

After repairing the database, index your files again.

## Tools Layer

Reusable app and agent actions live in `tools/`:

- `tools/indexing_tools.py` indexes uploaded documents and code folders.
- `tools/search_tools.py` searches relevant chunks and formats search context for agents.

Lower-level modules still stay where they belong: `loaders/` reads files, `rag/` handles chunks, embeddings, and ChromaDB, and `core/` handles config and LLM calls.
