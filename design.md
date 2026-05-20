# Design Document: Privacy-Preserving RAG Resume Matcher

## 1. Overview
A lightweight, privacy-focused RAG pipeline wrapped in a FastAPI REST API. It evaluates resumes against job descriptions using:
- **Local Sanitization**: CPU-optimized local models (via Ollama) for privacy.
- **Efficient Retrieval**: FAISS for in-memory, lightning-fast semantic search.
- **Frontier Reasoning**: Google Gemini (Free Tier) for high-level analysis and scoring.

## 2. Technical Stack (Minimal Bloat Approach)

| Component | Recommended Tool | Justification |
| :--- | :--- | :--- |
| **Framework** | `FastAPI` + `Uvicorn` | High performance, minimal overhead, built-in async support. |
| **Frontier LLM** | `Google Generative AI` SDK | Access to Gemini Pro/Flash; generous free tier. |
| **Local LLM** | `Ollama` (`Llama-3.2-3B` or `Phi-3.5-mini`) | Offloads processing to a local daemon; keeps Python env lean. |
| **Vector DB** | `FAISS` (cpu) | Industry standard for efficient, in-memory similarity search. |
| **Embeddings** | `fastembed` | Highly optimized for CPU (ONNX); significantly faster/lighter than `sentence-transformers`. |
| **PDF Parsing** | `pypdf` | Pure Python, lightweight, no external C-dependencies. |
| **Web Scraping** | `httpx` + `selectolax` | `selectolax` is much faster and uses less memory than BeautifulSoup. |

## 3. Detailed Pipeline Logic

### Phase 1: Resume Processing (Local & Private)
1. **Extraction**: Parse the Resume PDF using `pypdf` into meaningful, self-contained text chunks (e.g., 500-1000 characters).
2. **Sanitization (Local LLM via Ollama)**: Run each chunk through a local model (e.g., `Llama-3.2`) for:
    - **PII Redaction**: Identify and mask sensitive data (Names, Emails, Phone numbers). Use Regex for deterministic patterns first.
3. **Indexing**: Generate embeddings via `fastembed` and insert chunks into the **Resume FAISS Index** (in-memory).

### Phase 2: Job Description (JD) Processing
4. **Acquisition**: Fetch the JD from the provided source:
    - **PDF**: Parse with `pypdf`.
    - **URL**: Scrape using `httpx` and `selectolax`.
    - **Text**: Use direct input.
5. **Indexing**: Chunk the JD, generate embeddings, and insert into the **JD FAISS Index** (in-memory).

### Phase 3: Requirement Synthesis (Frontier)
6. **Requirement Analysis**: 
    - Perform a semantic search on the JD index to gather context.
    - Use **Google Gemini** to summarize the JD into a set of formalized requirements.
    - **Output Structure**: Each requirement MUST be formalized as a specific, answerable question about the candidate (e.g., "Does the candidate have 3+ years of experience with React?").

### Phase 3.5: Background-checking (Validation)
7. **Reference Extraction**: Extract concrete URL-like references (LinkedIn, GitHub, Portfolio) from the resume text.
8. **Automated Verification**: For up to **3 references**:
    - Fetch the content of the URL using `httpx` and `selectolax`.
    - Use **Google Gemini** to summarize the "Proof of Work" or "Claim Validation" found at these links relative to the requirements.
    - Store these summaries as "External Evidence."

### Phase 4: Matching & Evaluation (Frontier)
9. **Requirement Verification**: For each formalized question:
    - i) Perform a semantic search in the **Resume FAISS Index** to retrieve internal evidence chunks.
    - ii) Provide the **External Evidence** (from Phase 3.5) as additional grounding context.
    - iii) Use **Google Gemini** to check if the combined evidence confirms the requirement.
10. **Final Evaluation**: 
    - Gemini aggregates internal resume data and external background evidence into a fair, grounded summary.
    - **Scoring**: Compute a final match score (0-100). The score should be weighted higher if claims are verified by external links.
    - **Output**: Return a structured JSON response containing the summary, requirement list, verified links, and final score.

## 4. Why this stack?
- **Dependency Control**: By using `Ollama` as an external process, we avoid heavy local model weights and `torch`/`transformers` dependencies inside our Python app.
- **CPU Optimized**: `fastembed` and `FAISS-CPU` are designed to run fast on standard hardware without a GPU.
- **Cost**: Leveraging Gemini's free tier keeps the operational cost at $0.

## 5. REST API Endpoints (FastAPI)
- `POST /process-resume`: Upload PDF, sanitize, and store in session memory.
- `POST /process-jd`: Provide URL/PDF/Text, extract requirements.
- `GET /match-status`: Return the evaluation and final score.
