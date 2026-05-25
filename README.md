# `cvchecker`: a Privacy-Preserving RAG CV Matcher

## Overview
A lightweight, privacy-focused RAG pipeline wrapped in a FastAPI REST API. It evaluates CVs against job postings using:
- **Local Sanitization**: CPU-optimized local models (via Ollama) for privacy.
- **Efficient Retrieval**: FAISS for in-memory, lightning-fast semantic search.
- **Frontier Reasoning**: Google Gemini (Free Tier) for high-level analysis and scoring.

## Prerequisites

A quick note about the prerequisites...

- **Python 3.13** was used for the development of this project, so no guarantees are given that it will work with any other versions of Python (although it almost certainly will).
- Some parts of the pipeline depend on `ollama` running on the host where the API is deployed. Thus [`ollama`](https://ollama.com/) shall be installed on the host.
- Other parts rely on Google's Gemini models, thus an appropriate `GEMINI_API_KEY` variable shall be set in the environment to a key from [**Google AI studio**](https://aistudio.google.com/api-keys).
- To avoid hitting rate limits with [**HuggingFace**](https://huggingface.co/settings/tokens) (used for the embedding model and the test dataset), it's advisable to set the `HF_TOKEN` variable as well.

## Build & run

Ensure the `llama3.2:3b` model is downloaded to the host running Ollama:
```bash
ollama pull llama3.2:3b
```

> [!TIP]
> If you don't have Ollama running in the background, you can start it up in a separate terminal with `ollama serve`.

The [`uv`](https://docs.astral.sh/uv/) package manager is used for this project, thus to pull all the dependencies, you'll first have to:
```bash
uv sync
```

Next, to get the system up and running locally in a developer environment:
```bash
fastapi dev
```

...or if you want to use it in a production environment (not recommended as of now):
```bash
fastapi run
```

By default, the API is exposed on [http://localhost:8000/](http://localhost:8000/), but this can be configured to anything else (with the help of FastAPI) if needed.

## Technical Stack

| Component | Recommended Tool | Justification |
| :--- | :--- | :--- |
| **Framework** | `FastAPI` | High performance, minimal overhead, built-in async support. |
| **Frontier LLM** | `Google Generative AI` SDK | Access to Gemini Pro/Flash; generous free tier. |
| **Local LLM** | `Ollama` (`Llama-3.2-3B`) | Offloads processing to a local daemon; keeps Python env lean. |
| **Vector DB** | `FAISS` (cpu) | Industry standard for efficient, in-memory similarity search. |
| **Embeddings** | `fastembed` | Highly optimized for CPU (ONNX); significantly faster/lighter than `sentence-transformers`. |
| **PDF Parsing** | `pypdf` | Pure Python, lightweight, no external C-dependencies. |
| **Web Scraping** | `httpx` + `selectolax` | `selectolax` is much faster and uses less memory than BeautifulSoup. |
| **Misc.** | `pydantic` + `dotenv` | Needed to accomplish various tasks related to the main funtionality. |

## Detailed Pipeline Logic

### Phase 1: Resume Processing (Local & Private)
1. **Extraction**: Parse the Resume PDF using `pypdf` into meaningful, self-contained text chunks (e.g., 500-1000 characters).
2. **Sanitization (Local LLM via Ollama)**: Run each chunk through a local model (e.g., `Llama-3.2`) for:
    - **PII Redaction**: Identify and mask sensitive data (Names, Emails, Phone numbers). Use Regex for deterministic patterns first.
3. **Indexing**: Generate embeddings via `fastembed` and insert chunks into the **Resume FAISS Index** (in-memory).

### Phase 2: Job Posting (JP) Processing
4. **Acquisition**: Fetch the JP from the provided URL, and scrape it using `httpx` and `selectolax`.
5. **Indexing**: Chunk the JP, generate embeddings, and insert into the **JP FAISS Index** (in-memory).
6. **Requirement Analysis**: 
    - Perform a semantic search on the JP index to gather context.
    - Use **Google Gemini** to summarize the JP into a set of formalized requirements.
    - **Output Structure**: Each requirement MUST be formalized as a specific, answerable question about the candidate (e.g., "Does the candidate have 3+ years of experience with React?").

### Phase 3: Matching & Evaluation (Frontier)
7. **Requirement Verification**: For each formalized question:
    - i) Perform a semantic search in the **Resume FAISS Index** to retrieve internal evidence chunks.
    - ii) Provide the **External Evidence** (from Phase 3.5) as additional grounding context.
    - iii) Use **Google Gemini** to check if the combined evidence confirms the requirement.
8. **Final Evaluation**: 
    - Gemini aggregates evidences found in the CV into a fair, grounded summary.
    - **Scoring**: Compute a final match score (0-100). The score should be weighted higher if claims are verified by external links.
    - **Output**: Return a structured JSON response containing the summary, requirement list, verified links, and final score.

## REST API Endpoints
- `POST /check-cv`: Upload PDF, sanitize, do the processing, and generate a report.
