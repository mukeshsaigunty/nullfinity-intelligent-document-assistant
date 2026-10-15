
import json
import os
import urllib.parse
import boto3
from datetime import datetime, timezone
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract

s3 = boto3.client("s3")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")


def extract_pdf_text(pdf_bytes):
    """Extract selectable text first, then OCR scanned pages."""
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        method = "pdf_text"

        if not text.strip():
            images = convert_from_bytes(
                pdf_bytes,
                first_page=page_number,
                last_page=page_number,
                dpi=150,
            )
            text = pytesseract.image_to_string(images[0])
            method = "tesseract_ocr"

        pages.append({
            "page": page_number,
            "text": text.strip(),
            "method": method,
        })

    return pages


def lambda_handler(event, context):
    results = []

    for record in event.get("Records", []):
        if record.get("eventSource") != "aws:s3":
            continue

        source_bucket = record["s3"]["bucket"]["name"]
        source_key = urllib.parse.unquote_plus(
            record["s3"]["object"]["key"]
        )

        if not source_key.startswith("raw/"):
            continue

        if not source_key.lower().endswith(".pdf"):
            continue

        filename = source_key.split("/")[-1]
        output_bucket = OUTPUT_BUCKET or source_bucket

        try:
            response = s3.get_object(
                Bucket=source_bucket,
                Key=source_key,
            )
            pdf_bytes = response["Body"].read()

            pages = extract_pdf_text(pdf_bytes)

            text_key = f"processed/{filename}.txt"
            text_output = {
                "source_bucket": source_bucket,
                "source_key": source_key,
                "processed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "pages": pages,
            }

            s3.put_object(
                Bucket=output_bucket,
                Key=text_key,
                Body=json.dumps(
                    text_output, ensure_ascii=False, indent=2
                ).encode("utf-8"),
                ContentType="application/json",
            )

            results.append({
                "source": source_key,
                "output": text_key,
                "status": "OCR_COMPLETED",
                "pages": len(pages),
            })

        except Exception as exc:
            print(f"Processing failed for {source_key}: {exc}")
            raise

    return {
        "statusCode": 200,
        "body": json.dumps(results),
    }
