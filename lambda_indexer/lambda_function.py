
import json
import os
import re
import boto3
from sentence_transformers import SentenceTransformer

BUCKET = os.environ.get(
    "DOCUMENT_BUCKET",
    "nullfinity-mukesh-mumbai-2026"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

s3 = boto3.client("s3", region_name="ap-south-1")

# Load the model once per Lambda execution environment.
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def split_text(text):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def process_ocr_object(key):
    response = s3.get_object(
        Bucket=BUCKET,
        Key=key
    )

    document = json.loads(
        response["Body"].read().decode("utf-8")
    )

    source_key = document.get("source_key", key)
    document_name = os.path.basename(source_key)

    chunks = []

    for page_data in document.get("pages", []):
        page_number = page_data.get("page")
        page_text = page_data.get("text", "")

        for index, text in enumerate(split_text(page_text)):
            chunks.append({
                "chunk_id": (
                    f"{document_name}-page-{page_number}"
                    f"-chunk-{index + 1}"
                ),
                "document_name": document_name,
                "source_key": source_key,
                "page": page_number,
                "chunk_index": index,
                "text": text,
                "extraction_method": page_data.get("method")
            })

    if not chunks:
        print(f"No text found in {key}")
        return {"status": "NO_TEXT", "source": key}

    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True
    )

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector.tolist()

    output_key = f"vectors/{document_name}.json"

    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=json.dumps(
            chunks,
            ensure_ascii=False
        ).encode("utf-8"),
        ContentType="application/json"
    )

    print(f"Indexed {len(chunks)} chunks to {output_key}")

    return {
        "status": "INDEXED",
        "source": key,
        "output": output_key,
        "chunks": len(chunks)
    }


def lambda_handler(event, context):
    results = []

    for record in event.get("Records", []):
        bucket_name = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        # Ignore unrelated objects and avoid processing vectors.
        if not key.startswith("processed/"):
            continue

        if not key.endswith(".txt"):
            continue

        if bucket_name != BUCKET:
            print(f"Skipping unexpected bucket: {bucket_name}")
            continue

        results.append(process_ocr_object(key))

    return {
        "statusCode": 200,
        "body": json.dumps(results)
    }
