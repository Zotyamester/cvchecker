from pathlib import Path
import sys
from typing import BinaryIO, List, Optional

import faiss
from fastapi import FastAPI, HTTPException, UploadFile, status
from fastembed import TextEmbedding
from google import genai
from google.genai import types
from httpx import URL
import httpx
import numpy as np

from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from pydantic import BaseModel, Field, HttpUrl
from pypdf import PdfReader
from selectolax.lexbor import LexborHTMLParser
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_MODEL = "llama3.2:3b"
FRONTIER_MODEL = "gemini-3.5-flash"

embedding_model = TextEmbedding(model=EMBEDDING_MODEL)
gemini = genai.Client()

MAX_JOB_POSTING_CHUNKS = 10
MAX_REQUIREMENT_SATISFYING_CV_CHUNKS = 5

app = FastAPI()


def parse_pdf(file: BinaryIO) -> str:
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
    embeddings = np.array(list(embedding_model.embed(chunks)), dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings, faiss.Float32)
    return (chunks, index)


def process_cv(file: BinaryIO) -> tuple[list[str], faiss.IndexFlatL2]:
    parse = RunnableLambda(parse_pdf)
    redact = RunnableLambda(redact_pii_from_text)
    chunk = RunnableLambda(chunk_text)
    index = RunnableLambda(index_chunks)

    cv_processing_chain = parse | redact | chunk | index

    cv_chunks, cv_indices = cv_processing_chain.invoke(file)
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
    _, chunk_indices = index.search(query_embedding, MAX_JOB_POSTING_CHUNKS)
    chunks = "\n".join(f"<chunk>{raw_chunks[i]}</chunk>" for i in chunk_indices[0])

    response = gemini.models.generate_content(
        model=FRONTIER_MODEL,
        contents=f"""
        You are a hiring expert.
        Your task is to analyze short chunks of a job posting to formalize the posting (role and requirements) contained in the chunks.
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
    score: int = Field(
        description="A score from 0 to 100 that quantifies the suitability of the candidate for the role, based on the CV and the job posting. A score of 0 means that the candidate is not suitable at all for the role, while a score of 100 means that the candidate is perfectly suitable for the role. The score should be based on how well the candidate's qualifications, skills, and experiences (as described in the CV) match the requirements and criteria outlined in the job posting."
    )
    suitability: str = Field(
        description="A detailed explanation of how suitable the candidate is for the role, based on the CV and the job posting. The explanation should be as specific as possible, citing concrete examples from the CV that match the requirements of the job posting, and providing a justification for why those examples are relevant."
    )


class RequirementMatch(BaseModel):
    requirement: Requirement
    score: int = Field(
        description="A score from 0 to 100 that quantifies the suitability of the candidate for this specific requirement, based on the CV and the job posting. A score of 0 means that the candidate does not meet this requirement at all, while a score of 100 means that the candidate fully meets this requirement. The score should be based on how well the candidate's qualifications, skills, and experiences (as described in the CV) match this specific requirement outlined in the job posting."
    )
    reason: str = Field(
        description="The reason why the candidate meets or does not meet this specific requirement, based on the CV and the job posting. The explanation should be as specific as possible, citing concrete examples from the CV."
    )


def generate_report(
    cv_chunks: list[str], cv_indices: faiss.IndexFlatL2, job_posting: JobPosting
) -> Report:
    evidences = []
    for requirement in job_posting.requirements:
        query = f"""How suitable is a candidate that meets the following requirement for the role of {job_posting.role}?

        Requirement: {requirement.description}
        Years of experience required: {requirement.years_of_experience if requirement.years_of_experience is not None else "Not specified"}
        Preferred: {"Yes" if requirement.preferred else "No"}
        """

        query_embedding = np.array(list(embedding_model.embed(query)), dtype=np.float32)
        _, chunk_indices = cv_indices.search(
            query_embedding, MAX_REQUIREMENT_SATISFYING_CV_CHUNKS
        )
        chunks = "\n".join(f"<chunk>{cv_chunks[i]}</chunk>" for i in chunk_indices[0])

        response = gemini.models.generate_content(
            model=FRONTIER_MODEL,
            contents=f"""
            You are a hiring expert.
            Your task is to come to a conclusion on how suitable a candidate is for a specific requirement of a job posting, based on short chunks of the candidate's CV.
            Use the provided context (chunks) for assembling the response (i.e., to answer the query).

            Query: {query}

            Context:
            <chunks>
            {chunks}
            </chunks>
            """,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_schema=RequirementMatch,
            ),
        )

        requirement_match = response.parsed
        if requirement_match is None:
            raise Exception("Failed to parse the requirement match")
        evidences.append(requirement_match)

    suitability_explanations = "\n\n".join(
        f"""Requirement: {evidence.requirement.description}
        Years of experience required: {evidence.requirement.years_of_experience if evidence.requirement.years_of_experience is not None else "Not specified"}
        Preferred: {"Yes" if evidence.requirement.preferred else "No"}

        Suitability score for this requirement: {evidence.score}

        Explanation: {evidence.reason}
        """ for evidence in evidences
    )

    response = gemini.models.generate_content(
        model=FRONTIER_MODEL,
        contents=f"""
        You are a hiring expert.
        Your task is to generate a report on the suitability of a candidate for a job role, based on the candidate's CV and the job posting.

        The report should contain:
        1. The official title of the role.
        2. A score from 0 to 100 that quantifies the suitability of the candidate for the role, based on the CV and the job posting. A score of 0 means that the candidate is not suitable at all for the role, while a score of 100 means that the candidate is perfectly suitable for the role. The score should be based on how well the candidate's qualifications, skills, and experiences (as described in the CV) match the requirements and criteria outlined in the job posting.
        3. A detailed explanation of how suitable the candidate is for the role, based on the CV and the job posting. The explanation should be as specific as possible, citing concrete examples from the CV that match the requirements of the job posting, and providing a justification for why those examples are relevant.

        Use the following evidences to support your analysis:

        {suitability_explanations}

        Now generate the report.
        """,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_schema=Report,
        ),
    )

    report = response.parsed
    if report is None:
        raise Exception("Failed to parse the report")
    return report


def check_cv(cv: BinaryIO, job_posting_link: str):
    cv_chunks, cv_indices = process_cv(cv)
    job_posting = process_job_posting(job_posting_link)
    report = generate_report(cv_chunks, cv_indices, job_posting)
    return report


@app.post("/check-cv")
def check_cv_api(cv: UploadFile, job_posting_link: HttpUrl):
    if cv.content_type != "application/pdf" or not (cv.filename or "").lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is not a PDF"
        )

    report = check_cv(cv.file, job_posting_link.encoded_string())
    return report
