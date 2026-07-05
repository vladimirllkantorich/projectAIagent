# DevVault AI Project Description and Reflection

## Project Description

DevVault AI is a Streamlit-based CrewAI agent app for asking questions about uploaded documents and indexed code projects. It helps a user understand project structure, explain code, debug common issues, summarize documents, and inspect retrieved sources.

The problem it solves is simple: project and document context gets spread across many files. DevVault AI creates a searchable local knowledge base and uses agents to answer questions from that indexed context.

## Tools Used By The Agent

- Document reader: reads PDF, TXT, and DOCX files.
- Code reader: scans project folders and reads allowed source files.
- Chunking tool: splits long text into smaller chunks.
- Embedding tool: creates embeddings through OpenAI-compatible APIs or local fallback embeddings.
- ChromaDB vector search: retrieves relevant chunks.
- CrewAI crew: runs the answer agent, critic agent, and supervisor synthesis agent.
- LLM client: calls OpenAI API or LM Studio.

## Memory

The app keeps recent chat messages in Streamlit session state. Recent conversation memory is passed into the agent pipeline as follow-up context. It is explicitly marked as background memory, not as source evidence.

## Error Handling

The project handles common failure cases:

- Missing OpenAI API key.
- LM Studio server not reachable.
- No local model loaded.
- ChromaDB database mismatch or corruption.
- Missing or unsupported uploaded document types.
- Embedding API unavailable, with fallback local hash embeddings.

## What Worked

- Streamlit made the UI simple to build and test.
- ChromaDB provided useful local retrieval over documents and code.
- CrewAI made the answer, critique, and synthesis workflow easier to describe as a formal multi-agent crew.
- Keeping RAG search deterministic made the local LM Studio mode more predictable than relying on model-driven tool calling.
- Code intent detection became more useful after adding formats for explanation, debugging, refactoring, architecture, imports, and RAG/project questions.

## What Did Not Work Perfectly

- ChromaDB can print telemetry warnings even when the app works.
- Retrieval quality depends heavily on whether the right files were indexed.
- If chunks are incomplete, the LLM must clearly say what context is missing.
- CrewAI requires Python 3.10-3.13, so the old Python 3.9 environment is no longer suitable.
- The project is a Streamlit app, not a single notebook, so submission may need these separate documentation files.

## What I Learned

- A useful CrewAI app can still keep deterministic tools outside the LLM when reliability matters.
- Tools should be small, direct actions such as reading files, indexing text, searching the vector store, and calling the LLM.
- Strong system prompts matter because they keep the model grounded.
- Memory is useful for follow-up questions, but retrieved sources should still be the main evidence.
- Code agents need different answer formats for explain, debug, refactor, architecture, and import questions.
