
import os
import tempfile

import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

POPPLER_PATH = (
    r"C:\Users\G.MUKESH SAI\Downloads"
    r"\Release-26.09.0-0\poppler-26.09.0\Library\bin"
)


def extract_text_from_pdf(uploaded_file):
    """Extract digital text, using OCR when a page has no text."""
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()

    pages = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        temp.write(pdf_bytes)
        temp_pdf_path = temp.name

    try:
        reader = PdfReader(temp_pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                images = convert_from_path(
                    temp_pdf_path,
                    first_page=page_number,
                    last_page=page_number,
                    poppler_path=POPPLER_PATH,
                    dpi=200,
                )

                if images:
                    text = pytesseract.image_to_string(
                        images[0], lang="eng"
                    ).strip()

            pages.append({
                "page_number": page_number,
                "text": text,
            })

    finally:
        os.remove(temp_pdf_path)

    return pages


def split_text(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def create_document_chunks(pages, document_name):
    all_chunks = []

    for page in pages:
        for chunk_index, chunk_text in enumerate(
            split_text(page["text"])
        ):
            page_number = page["page_number"]

            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "document_name": document_name,
                    "source": document_name,
                    "page_number": page_number,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "chunk_number": chunk_index + 1,
                },
            })

    return all_chunks
