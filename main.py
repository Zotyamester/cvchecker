from pathlib import Path
import sys
from typing import List, Optional

import faiss
from fastembed import TextEmbedding
from google import genai
from google.genai import types
from httpx import URL
import httpx
import numpy as np

from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from pydantic import BaseModel, Field
from pypdf import PdfReader
from selectolax.lexbor import LexborHTMLParser
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_MODEL = "llama3.2:3b"
FRONTIER_MODEL = "gemini-3.5-flash"

embedding_model = TextEmbedding(model=EMBEDDING_MODEL)
gemini = genai.Client()

MAX_JOB_posting_CHUNKS = 10


def parse_pdf(file: Path) -> str:
    """Read all textual content of a PDF File

    Args:
        file (PathLike): path to the input file

    Returns:
        str: extracted text
    """
    content = ""
    with PdfReader(file) as reader:
        for page in reader.pages:
            text = page.extract_text()
            content += text + "\n\n"
    return content


def redact_pii_from_text(text: str) -> str:
    result = ollama.generate(
        model=LOCAL_MODEL,
        stream=False,
        prompt=f"""Redact all Personally Identifiable Information (PII) from the following text.
    Only output the final, redacted text, with nothing before and after.
    Pay special attention to the name, phone, email, address, and other identification numbers.
    Redact PIIs by substituting them with a placeholder describing the kind of PII enclosed in brackets.
    For example, if the text is "John Doe (phone: +1 (234) 567-8901, email: john.doe@example.com) lives under 123 Main St, Anytown, USA",
    the redacted text shoud be "[NAME] (phone: [PHONE], email: [EMAIL]) lives under [ADDRESS]".
    Do NOT redact information related to education, prior work experience, skills, or any other information that is not PII.

    Now redact the following.
    
    Text:
    {text}

    Redacted:
""",
        options={"temperature": 0.1},
    )

    return result["response"]


def chunk_text(text: str) -> list[str]:
    """Split a corpus of text into small chunks

    Args:
        text (str): text to split

    Returns:
        list[str]: chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text)]


def index_chunks(chunks: list[str]) -> tuple[list[str], faiss.IndexFlatL2]:
    print("Indexing chunks...")
    embeddings = np.array(list(embedding_model.embed(chunks)), dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings, faiss.Float32)
    print("Done indexing.")
    return (chunks, index)


def process_cv(file: Path) -> tuple[list[str], faiss.IndexFlatL2]:
    parse = RunnableLambda(parse_pdf)
    redact = RunnableLambda(redact_pii_from_text)
    chunk = RunnableLambda(chunk_text)
    index = RunnableLambda(index_chunks)

    cv_processing_chain = (
        parse
        # | redact
        | chunk
        | index
    )

    cv_chunks, cv_indices = cv_processing_chain.invoke(cv_path)
    return (cv_chunks, cv_indices)


def retrieve_web_content(url: URL) -> str:
    r = httpx.get(url, follow_redirects=True)
    if not r.is_success:
        raise Exception("Failed to load job description")
    parser = LexborHTMLParser(r.content)

    # Heuristic for removing (most commonly) irrelevant nodes
    for node in parser.body.select("nav, header, footer, aside, script, style").matches:
        node.remove()

    text = parser.body.text(separator=" ", strip=True, skip_empty=True)

    return text


class Requirement(BaseModel):
    description: str = Field(
        description="Elaboration on the concrete requirement. For example, what skills, degrees, prior knowledge, or work experience are required for the role."
    )
    years_of_experience: Optional[int] = Field(
        description="How many years of experience are required in the described skills, if applicable?"
    )
    preferred: bool


class JobPosting(BaseModel):
    role: str = Field(description="Official title of the role in the job posting.")
    requirements: List[Requirement]


def formalize_job_posting(
    raw_chunks: list[str], index: faiss.IndexFlatL2
) -> JobPosting:
    query = "What is this role? Who is needed for this position? What criteria or requirements must be met by a candidate to be accepted for this job?"

    query_embedding = np.array(list(embedding_model.embed(query)), dtype=np.float32)
    _, chunk_indices = index.search(query_embedding, MAX_JOB_posting_CHUNKS)
    chunks = "\n".join(f"<chunk>{raw_chunks[i]}</chunk>" for i in chunk_indices[0])

    response = gemini.models.generate_content(
        model=FRONTIER_MODEL,
        contents=f"""
        You are a hiring expert.
        Your task is to analyze short chunks of a job posting to make an internal report that formalizes the posting (role and requirements) contained in the chunks.
        Use the provided context (chunks) for assembling the response (i.e., to answer the query).

        Query: {query}

        Context:
        <chunks>
        {chunks}
        </chunks>
        """,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_schema=JobPosting,
        ),
    )

    job_posting = response.parsed
    if job_posting is None:
        raise Exception("Failed to parse the job posting")

    return job_posting


def process_job_posting(url: str) -> JobPosting:
    retrieve = RunnableLambda(retrieve_web_content)
    chunk = RunnableLambda(chunk_text)
    index = RunnableLambda(index_chunks)

    job_posting_processing_chain = retrieve | chunk | index

    job_posting_chunks, job_posting_indices = job_posting_processing_chain.invoke(url)
    job_posting = formalize_job_posting(job_posting_chunks, job_posting_indices)
    return job_posting


class Report(BaseModel):
    role: str = Field(description="The official title of the role.")
    suitability: str = Field(
        description="A detailed explanation of how suitable the candidate is for the role, based on the CV and the job posting. The explanation should be as specific as possible, citing concrete examples from the CV that match the requirements of the job posting, and providing a justification for why those examples are relevant."
    )
    score: int = Field(
        description="A score from 0 to 100 that quantifies the suitability of the candidate for the role, based on the CV and the job posting. A score of 0 means that the candidate is not suitable at all for the role, while a score of 100 means that the candidate is perfectly suitable for the role. The score should be based on how well the candidate's qualifications, skills, and experiences (as described in the CV) match the requirements and criteria outlined in the job posting."
    )


def generate_report(
    cv_chunks: list[str], cv_indices: faiss.IndexFlatL2, job_posting: JobPosting
) -> Report: ...


def main():
    cv_path = Path(sys.argv[1])
    job_posting_url = URL(sys.argv[2])

    cv_chunks, cv_indices = process_cv(cv_path)
    job_posting = process_job_posting(job_posting_url)
    report = generate_report(cv_chunks, cv_indices, job_posting)
    print(report.json(indent=4))


if __name__ == "__main__":
    main()
