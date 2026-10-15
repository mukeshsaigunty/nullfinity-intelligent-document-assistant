
import json
import re
import os
import boto3
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
BUCKET = "nullfinity-mukesh-mumbai-2026"

OCR_KEY = "processed/automatic-test.pdf.txt"

OUTPUT_FILE = "embedded_chunks.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

AWS_REGION = "ap-south-1"

s3 = boto3.client("s3", region_name=AWS_REGION)


# --------------------------------------------------
# 2. Split text into chunks
# --------------------------------------------------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if chunk_size <= 0 or overlap < 0:
        raise ValueError("Chunk size must be positive and overlap non-negative.")

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# --------------------------------------------------
# 3. Load OCR output from S3
# --------------------------------------------------
def load_ocr_document():
    print(f"Downloading OCR output: s3://{BUCKET}/{OCR_KEY}")

    response = s3.get_object(
        Bucket=BUCKET,
        Key=OCR_KEY
    )

    document = json.loads(
        response["Body"].read().decode("utf-8")
    )

    print("Source document:", document.get("source_key", OCR_KEY))
    print("Pages found:", len(document.get("pages", [])))

    return document


# --------------------------------------------------
# 4. Create chunks with metadata
# --------------------------------------------------
def create_chunks(document):
    all_chunks = []

    source_key = document.get("source_key", OCR_KEY)

    # raw/invoice.pdf becomes invoice.pdf
    document_name = os.path.basename(source_key)

    for page_data in document.get("pages", []):
        page_number = page_data.get("page")
        page_text = page_data.get("text", "")

        page_chunks = split_text(page_text)

        for index, text in enumerate(page_chunks):
            all_chunks.append({
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

    return all_chunks


# --------------------------------------------------
# 5. Generate embeddings
# --------------------------------------------------
def generate_embeddings(all_chunks, model):
    if not all_chunks:
        print("No text found to embed.")
        return all_chunks

    texts = [chunk["text"] for chunk in all_chunks]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    for chunk, vector in zip(all_chunks, vectors):
        chunk["embedding"] = vector.tolist()

    print("\nEmbedding generation successful!")
    print("Vector dimensions:", len(all_chunks[0]["embedding"]))
    print("Chunks with embeddings:", len(all_chunks))

    return all_chunks


# --------------------------------------------------
# 6. Save locally and upload index to S3
# --------------------------------------------------
def save_embeddings(all_chunks):
    # Keep a local copy for debugging
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved embeddings locally to {OUTPUT_FILE}")

    if not all_chunks:
        print("No embeddings to upload.")
        return

    # Use the original PDF filename to avoid a shared hardcoded key.
    document_name = all_chunks[0]["document_name"]
    output_key = f"vectors/{document_name}.json"

    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=json.dumps(
            all_chunks,
            ensure_ascii=False
        ).encode("utf-8"),
        ContentType="application/json"
    )

    print(f"Uploaded embeddings to s3://{BUCKET}/{output_key}")


# --------------------------------------------------
# 7. Search chunks using cosine similarity
# --------------------------------------------------
def search_chunks(question, all_chunks, model, top_k=TOP_K):
    if not all_chunks:
        print("No chunks available for searching.")
        return []

    question_vector = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    results = []

    for chunk in all_chunks:
        if "embedding" not in chunk:
            continue

        chunk_vector = np.asarray(
            chunk["embedding"],
            dtype=np.float32
        )

        # Normalized vectors: dot product equals cosine similarity.
        similarity = float(np.dot(question_vector, chunk_vector))

        results.append({
            "similarity": similarity,
            "document_name": chunk["document_name"],
            "source_key": chunk["source_key"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"]
        })

    results.sort(
        key=lambda result: result["similarity"],
        reverse=True
    )

    return results[:top_k]


# --------------------------------------------------
# 8. Main execution
# --------------------------------------------------
def main():
    document = load_ocr_document()

    all_chunks = create_chunks(document)

    print("\nTotal chunks:", len(all_chunks))

    if not all_chunks:
        print("No chunks were created. Stopping.")
        return

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    all_chunks = generate_embeddings(all_chunks, model)

    save_embeddings(all_chunks)

    # Test semantic retrieval
    question = "What is the invoice total?"

    print("\nQuestion:", question)
    print("\nMost relevant chunks:")

    results = search_chunks(
        question=question,
        all_chunks=all_chunks,
        model=model,
        top_k=TOP_K
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print("Similarity:", round(result["similarity"], 4))
        print("Document:", result["document_name"])
        print("Page:", result["page"])
        print("Chunk ID:", result["chunk_id"])
        print("Text:", result["text"])


if __name__ == "__main__":
    main()
