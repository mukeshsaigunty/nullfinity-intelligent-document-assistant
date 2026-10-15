
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the Hugging Face sentence embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def generate_embeddings(texts, model):
    """Convert text chunks into vector embeddings."""
    if not texts:
        return []

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()
