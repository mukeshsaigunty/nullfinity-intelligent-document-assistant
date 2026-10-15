
import hashlib
from pathlib import Path

import chromadb

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "pdf_documents"


def get_collection():
    """Create or load the persistent ChromaDB collection."""
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(chunks, embeddings):
    """Store document chunks, embeddings, and metadata."""
    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for item in chunks:
        text = item["text"]
        metadata = item["metadata"]

        # Stable ID: re-indexing the same chunk updates it.
        identifier = (
            f"{metadata['document_name']}_"
            f"{metadata['page_number']}_"
            f"{metadata['chunk_index']}"
        )
        chunk_id = hashlib.sha256(
            identifier.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(metadata)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search_documents(query_embedding, top_k=4):
    """Retrieve the most relevant document chunks."""
    collection = get_collection()

    total = collection.count()
    if total == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for i, text in enumerate(results["documents"][0]):
        matches.append({
            "text": text,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return matches
