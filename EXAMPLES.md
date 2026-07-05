# DevVault AI Usage Examples

These examples are written for the AI Agents project submission. The exact wording of the output can change depending on the selected model and the indexed files, but the behavior should follow this shape.

## Example 1: Ask About Uploaded Documents

Input:

```text
Summarize the main idea of the uploaded PDF.
```

Expected output:

```text
The document explains the main topic in a few key points...

Sources:
- uploaded-file.pdf (3 chunks)
```

Explanation:

DevVault AI uses the document-reading tool to extract text, splits it into chunks, stores it in ChromaDB, retrieves the most relevant chunks, and asks the CrewAI Document Answer Agent to answer from those chunks.

## Example 2: Ask About Code

Input:

```text
Explain how the SupervisorAgent chooses which agent should answer.
```

Expected output:

```text
What this code does
- SupervisorAgent coordinates the answer flow.

Main flow
- TaskAgent classifies the question before the CrewAI tasks are built.
- SearchAgent retrieves relevant chunks.
- SupervisorAgent starts a CrewAI crew with an answer task, critique task, and final synthesis task.

Important functions/classes
- SupervisorAgent.answer
- CrewAIRagPipeline.answer
- TaskAgent.plan

Possible problems
- If the indexed chunks do not include the relevant file, the answer may be incomplete.

Suggested improvements
- Keep the routing logic small and easy to read.

Sources:
- agents/supervisor.py (2 chunks)
- agents/task_agent.py (1 chunks)
```

Explanation:

TaskAgent classifies the question as code-related. SearchAgent finds relevant code chunks. The CrewAI Code Analysis Agent receives the code-question format and returns a structured answer grounded in the retrieved code.

## Example 3: Debug A Code Issue

Input:

```text
Why does this import fail in CodeAgent?
```

Expected output:

```text
Likely cause
- The imported name may not exist in the module, or the module path may be wrong.

Where to check
- agents/code_agent.py
- tools/search_tools.py

Fix
- Confirm that format_search_context is exported from tools/search_tools.py.
- Confirm that the import path uses tools.search_tools.

Why this fix works
- Python imports names from the module listed in the import statement. If the name exists there, the import succeeds.

Sources:
- agents/code_agent.py (1 chunks)
- tools/search_tools.py (1 chunks)
```

Explanation:

The CrewAI Code Analysis Agent receives a debug/import-style answer format, uses the retrieved code chunks, and gives a practical debugging path instead of guessing about files that were not retrieved.
