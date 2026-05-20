from pathlib import Path
import sys

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


def main():
    text = parse_pdf(Path(sys.argv[1]))
    redacted = redact_pii_from_text(text)
    chunks = chunk_text(redacted)

    for chunk in chunks:
        print(f"<chunk>\n{chunk}\n<chunk>", end="\n\n")


if __name__ == "__main__":
    main()
