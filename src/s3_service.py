
import os
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# -------------------- Configuration --------------------

MUMBAI_BUCKET = os.getenv(
    "MUMBAI_S3_BUCKET",
    "nullfinity-mukesh-mumbai-2026",
)
MUMBAI_REGION = "ap-south-1"

OLD_BUCKET = os.getenv(
    "OLD_S3_BUCKET",
    "nullfinity-mukesh",
)
OLD_REGION = "ap-south-2"


# -------------------- S3 Clients --------------------

def get_s3_client():
    """Create an S3 client for the primary Mumbai bucket."""
    return boto3.client("s3", region_name=MUMBAI_REGION)


def get_old_s3_client():
    """Create an S3 client for the old Hyderabad bucket."""
    return boto3.client("s3", region_name=OLD_REGION)


# -------------------- Upload PDF to Mumbai --------------------

def upload_pdf(file_obj, filename):
    """Upload a PDF to Mumbai S3 under raw/."""
    s3 = get_s3_client()
    safe_filename = Path(filename).name
    object_key = f"raw/{safe_filename}"

    try:
        file_obj.seek(0)
        s3.upload_fileobj(
            file_obj,
            MUMBAI_BUCKET,
            object_key,
            ExtraArgs={"ContentType": "application/pdf"},
        )

        return {
            "success": True,
            "bucket": MUMBAI_BUCKET,
            "region": MUMBAI_REGION,
            "key": object_key,
            "uri": f"s3://{MUMBAI_BUCKET}/{object_key}",
        }

    except ClientError as exc:
        error = exc.response.get("Error", {})
        raise RuntimeError(
            f"Mumbai S3 upload failed: "
            f"{error.get('Code', 'Unknown')} - "
            f"{error.get('Message', str(exc))}"
        ) from exc


# -------------------- Save Extracted Text to Mumbai --------------------

def save_extracted_text(document_name, pages):
    """Save extracted PDF/OCR text to processed/ in Mumbai S3."""
    safe_filename = Path(document_name).name
    object_key = f"processed/{safe_filename}.txt"

    page_texts = []

    for page in pages:
        page_number = page.get(
            "page_number",
            page.get("page", "Unknown"),
        )
        text = page.get("text", "")
        page_texts.append(f"--- Page {page_number} ---\n{text}")

    extracted_text = "\n\n".join(page_texts)

    try:
        get_s3_client().put_object(
            Bucket=MUMBAI_BUCKET,
            Key=object_key,
            Body=extracted_text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
            Metadata={
                "original-filename": safe_filename,
                "extraction-method": "local-pdf-text-and-ocr",
            },
        )

        return {
            "success": True,
            "bucket": MUMBAI_BUCKET,
            "region": MUMBAI_REGION,
            "key": object_key,
            "uri": f"s3://{MUMBAI_BUCKET}/{object_key}",
            "characters": len(extracted_text),
        }

    except ClientError as exc:
        error = exc.response.get("Error", {})
        raise RuntimeError(
            f"Saving extracted text to Mumbai S3 failed: "
            f"{error.get('Code', 'Unknown')} - "
            f"{error.get('Message', str(exc))}"
        ) from exc


# -------------------- List Mumbai PDFs --------------------

def list_s3_pdfs():
    """List PDFs stored under raw/ in Mumbai."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    pdf_files = []

    try:
        for page in paginator.paginate(
            Bucket=MUMBAI_BUCKET,
            Prefix="raw/",
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                if key.lower().endswith(".pdf"):
                    pdf_files.append({
                        "name": Path(key).name,
                        "key": key,
                        "size": obj["Size"],
                        "size_bytes": obj["Size"],
                        "last_modified": obj["LastModified"],
                    })

        return pdf_files

    except ClientError as exc:
        error = exc.response.get("Error", {})
        raise RuntimeError(
            f"Could not list Mumbai S3 files: "
            f"{error.get('Code', 'Unknown')} - "
            f"{error.get('Message', str(exc))}"
        ) from exc


# -------------------- List Old Hyderabad PDFs --------------------

def list_old_bucket_pdfs():
    """List PDFs stored under raw/ in the old Hyderabad bucket."""
    s3 = get_old_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    pdf_files = []

    try:
        for page in paginator.paginate(
            Bucket=OLD_BUCKET,
            Prefix="raw/",
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                if key.lower().endswith(".pdf"):
                    pdf_files.append({
                        "name": Path(key).name,
                        "key": key,
                        "size": obj["Size"],
                        "size_bytes": obj["Size"],
                        "last_modified": obj["LastModified"],
                    })

        return pdf_files

    except ClientError as exc:
        error = exc.response.get("Error", {})
        raise RuntimeError(
            f"Could not list Hyderabad S3 files: "
            f"{error.get('Code', 'Unknown')} - "
            f"{error.get('Message', str(exc))}"
        ) from exc


# -------------------- Copy Mumbai PDF to Hyderabad --------------------

def copy_pdf_to_old_bucket(mumbai_key):
    """Copy a PDF from Mumbai to Hyderabad when requested."""
    source = {
        "Bucket": MUMBAI_BUCKET,
        "Key": mumbai_key,
    }
    destination_key = f"raw/{Path(mumbai_key).name}"

    try:
        get_old_s3_client().copy(
            source,
            OLD_BUCKET,
            destination_key,
        )

        return {
            "source_bucket": MUMBAI_BUCKET,
            "source_key": mumbai_key,
            "destination_bucket": OLD_BUCKET,
            "destination_key": destination_key,
        }

    except ClientError as exc:
        error = exc.response.get("Error", {})
        raise RuntimeError(
            f"Copy to Hyderabad failed: "
            f"{error.get('Code', 'Unknown')} - "
            f"{error.get('Message', str(exc))}"
        ) from exc


# -------------------- Download Processed Text --------------------

def download_text(object_key):
    """Download processed text from Mumbai S3."""
    try:
        response = get_s3_client().get_object(
            Bucket=MUMBAI_BUCKET,
            Key=object_key,
        )
        return response["Body"].read().decode("utf-8")

    except ClientError as exc:
        error = exc.response.get("Error", {})
        raise RuntimeError(
            f"Mumbai S3 download failed: "
            f"{error.get('Code', 'Unknown')} - "
            f"{error.get('Message', str(exc))}"
        ) from exc


# -------------------- Load S3 Vector Index --------------------

def load_s3_vector_chunks():
    """
    Load embedded chunks created by nullfinity-document-indexer.

    Reads JSON files from vectors/ and preserves:
    - Document filename
    - Source S3 key
    - Original page number
    - Chunk text
    - Embedding vector
    """
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    chunks = []

    try:
        for page in paginator.paginate(
            Bucket=MUMBAI_BUCKET,
            Prefix="vectors/",
        ):
            for obj in page.get("Contents", []):
                object_key = obj["Key"]

                if not object_key.lower().endswith(".json"):
                    continue

                response = s3.get_object(
                    Bucket=MUMBAI_BUCKET,
                    Key=object_key,
                )

                try:
                    payload = json.loads(
                        response["Body"].read().decode("utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"Invalid JSON in {object_key}: {exc}"
                    ) from exc

                if not isinstance(payload, list):
                    continue

                for item in payload:
                    if not isinstance(item, dict):
                        continue

                    text = item.get("text")
                    embedding = item.get("embedding")

                    if (
                        not isinstance(text, str)
                        or not text.strip()
                        or not isinstance(embedding, list)
                        or not embedding
                    ):
                        continue

                    source_key = item.get("source_key", "")
                    document_name = (
                        item.get("document_name")
                        or Path(source_key).name
                        or Path(object_key).stem
                    )

                    chunk = item.copy()
                    chunk["text"] = text
                    chunk["embedding"] = embedding
                    chunk["document_name"] = document_name
                    chunk["page"] = item.get(
                        "page",
                        item.get("page_number", "Unknown"),
                    )
                    chunk["source_key"] = source_key
                    chunk["vector_s3_key"] = object_key

                    chunks.append(chunk)

        return chunks

    except ClientError as exc:
        error = exc.response.get("Error", {})
        raise RuntimeError(
            f"Failed to load vectors from Mumbai S3: "
            f"{error.get('Code', 'Unknown')} - "
            f"{error.get('Message', str(exc))}"
        ) from exc


# -------------------- Test Connections --------------------

def test_s3_connection():
    """Check access to both S3 buckets."""
    results = {}

    for name, client, bucket in [
        ("Mumbai", get_s3_client(), MUMBAI_BUCKET),
        ("Hyderabad", get_old_s3_client(), OLD_BUCKET),
    ]:
        try:
            client.head_bucket(Bucket=bucket)
            results[name] = f"Connected: {bucket}"

        except ClientError as exc:
            error = exc.response.get("Error", {})
            results[name] = (
                f"Failed: {error.get('Code', 'Unknown')} - "
                f"{error.get('Message', 'Check IAM permissions')}"
            )

    return results
