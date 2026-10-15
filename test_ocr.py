
from pathlib import Path
from src.pdf_processor import extract_text_from_pdf

# Put your scanned PDF inside the data folder.
pdf_path = Path("data/ASSIGNMENT.pdf")

if not pdf_path.exists():
    print(f"PDF not found: {pdf_path}")
    print("Place a scanned PDF in the data folder and update its filename.")
else:
    with pdf_path.open("rb") as pdf_file:
        pages = extract_text_from_pdf(pdf_file)

    total_chars = 0

    for page in pages:
        text = page["text"]
        total_chars += len(text)
        print(f"\n--- Page {page['page_number']} ---")
        print(text[:1000] if text else "No text extracted")

    print(f"\nTotal extracted characters: {total_chars}")

    if total_chars > 0:
        print("SUCCESS: Text extraction/OCR returned text.")
    else:
        print("WARNING: No text extracted.")
