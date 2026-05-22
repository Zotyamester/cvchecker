from pathlib import Path
import sys


from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from pypdf import PdfReader

REDATION_MODEL = "llama3.2"


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
        model=REDATION_MODEL,
        prompt=f"""Redact all Personally Identifiable Information (PII) from the following text.
    Only output the final, redacted text, with nothing before and after.
    Pay special attention to name, phone, email, address, company, school, and other identification numbers.
    Redact PIIs by substituting them with a placeholder describing the kind of PII enclosed in brackets.
    For example, if the text is "John Doe (phone: +1 (234) 567-8901), graduated from Stanford University, and used to work for ABC, Corp.",
    the redacted text shoud be "[NAME] (phone: [PHONE]), graduated from [SCHOOL], and used to work for [COMPANY]".

    Now redact the following.
    
    Text:
    {text}

    Redacted:
""",
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


def index_chunks(chunks: list[str]) -> faiss.IndexFlatL2:
    # Import necessary library

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token=HF_TOKEN)
    chunk_embeddings = embedding_model
    IndexFlatL2


def main():
    cv_path = Path(sys.argv[1])

    parse = RunnableLambda(parse_pdf)
    redact = RunnableLambda(redact_pii_from_text)
    chunk = RunnableLambda(chunk_text)

    chain = parse | redact | chunk


if __name__ == "__main__":
    main()
