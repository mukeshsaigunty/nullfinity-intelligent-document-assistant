
from io import BytesIO

import pandas as pd
import streamlit as st

from src.rag_pipeline import answer_question
from src.s3_service import (
    MUMBAI_BUCKET,
    get_s3_client,
    upload_pdf,
    list_s3_pdfs,
    list_old_bucket_pdfs,
    load_s3_vector_chunks,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Nullfinity - Intelligent Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Nullfinity — Intelligent Document Assistant")

st.write(
    "Upload PDFs to Amazon S3, automatically extract text using "
    "AWS Lambda and OCR, generate embeddings, and ask questions "
    "with document and page citations."
)

st.info(
    "Architecture: S3 raw/ → OCR Lambda → processed/ → "
    "Embedding Lambda → vectors/ → RAG retrieval → Groq answer generation."
)


# ============================================================
# S3 helper functions
# ============================================================

def list_s3_objects(prefix, suffix=None):
    """List objects in the configured Mumbai S3 bucket."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    objects = []

    for page in paginator.paginate(
        Bucket=MUMBAI_BUCKET,
        Prefix=prefix,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if suffix and not key.lower().endswith(suffix.lower()):
                continue

            objects.append({
                "Filename": key.rsplit("/", 1)[-1],
                "S3 key": key,
                "Size (bytes)": obj["Size"],
                "Last modified": str(obj["LastModified"]),
            })

    return objects


def show_object_table(objects, empty_message):
    """Display a list of S3 objects."""
    if objects:
        st.dataframe(
            pd.DataFrame(objects),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(empty_message)


# ============================================================
# 1. Upload documents
# ============================================================

st.divider()
st.header("1. Upload and process PDF documents")

st.write(
    "Select PDFs to upload them to the raw/ folder in Mumbai S3. "
    "The configured S3 event triggers will start OCR and embedding "
    "generation automatically."
)

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
)

if st.button("Upload PDFs to AWS S3", type="primary"):
    if not uploaded_files:
        st.warning("Please select at least one PDF.")
    else:
        successful_uploads = 0
        failed_uploads = 0

        progress = st.progress(0)
        status = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                pdf_bytes = uploaded_file.getvalue()

                if not pdf_bytes:
                    st.warning(
                        f"{uploaded_file.name} is empty. Skipping."
                    )
                    failed_uploads += 1
                    continue

                status.info(
                    f"Uploading {uploaded_file.name}..."
                )

                result = upload_pdf(
                    BytesIO(pdf_bytes),
                    uploaded_file.name,
                )

                successful_uploads += 1

                st.success(
                    f"Uploaded {uploaded_file.name} to "
                    f"{result['uri']}"
                )

            except Exception as exc:
                failed_uploads += 1
                st.error(
                    f"Upload failed for {uploaded_file.name}: {exc}"
                )

            finally:
                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

        status.empty()
        progress.empty()

        st.subheader("Upload summary")

        col1, col2 = st.columns(2)
        col1.metric("Successful uploads", successful_uploads)
        col2.metric("Failed uploads", failed_uploads)

        if successful_uploads:
            st.success(
                "Upload complete. OCR and indexing run asynchronously. "
                "Use the refresh button below to check their progress."
            )
            st.warning(
                "Uploading a PDF with an existing filename overwrites "
                "the object under raw/. Use unique filenames for "
                "different documents."
            )


# ============================================================
# 2. AWS processing status
# ============================================================

st.divider()
st.header("2. AWS document-processing status")

st.write(
    "Refresh this section after uploading documents to check "
    "whether OCR and embedding generation have completed."
)

if st.button("Refresh AWS processing status"):
    try:
        with st.spinner("Checking S3 files..."):
            raw_files = list_s3_pdfs()
            processed_files = list_s3_objects(
                prefix="processed/",
                suffix=".txt",
            )
            vector_files = list_s3_objects(
                prefix="vectors/",
                suffix=".json",
            )

        col1, col2, col3 = st.columns(3)

        col1.metric("Original PDFs", len(raw_files))
        col2.metric("Processed OCR files", len(processed_files))
        col3.metric("Vector index files", len(vector_files))

        st.subheader("Original PDFs in raw/")
        show_object_table(
            raw_files,
            "No PDFs found under raw/.",
        )

        st.subheader("OCR outputs in processed/")
        show_object_table(
            processed_files,
            "No OCR outputs found yet. Upload a PDF and wait for the OCR Lambda.",
        )

        st.subheader("Embedding outputs in vectors/")
        show_object_table(
            vector_files,
            "No vector files found yet. Wait for the indexing Lambda.",
        )

    except Exception as exc:
        st.error(
            f"Could not retrieve AWS processing status: {exc}"
        )


# ============================================================
# 3. Ask questions using S3 vectors
# ============================================================

st.divider()
st.header("3. Ask questions about your documents")

st.write(
    "The RAG pipeline retrieves chunks from vectors/ in S3, "
    "combines semantic search with BM25, reranks the results, "
    "and asks Groq to generate an answer grounded in the retrieved text."
)

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is the total invoice amount?",
)

top_k = st.slider(
    "Number of final chunks to retrieve",
    min_value=1,
    max_value=8,
    value=4,
)

if st.button("Get answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner(
                "Retrieving S3 chunks, reranking, and generating an answer..."
            ):
                result = answer_question(
                    question=question.strip(),
                    top_k=top_k,
                )

            st.subheader("Answer")
            st.write(
                result.get(
                    "answer",
                    "No answer returned.",
                )
            )

            sources = result.get("sources", [])

            st.subheader("Retrieved sources")

            if not sources:
                st.info(
                    "No source passages were returned. Confirm that "
                    "OCR and embedding generation have completed."
                )

            for source in sources:
                document = source.get(
                    "document",
                    "Unknown document",
                )
                page = source.get("page", "Unknown")

                with st.expander(
                    f"📄 {document} — Page {page}"
                ):
                    st.write(source.get("text", ""))

                    if "rrf_score" in source:
                        st.caption(
                            f"RRF score: {source['rrf_score']}"
                        )

                    if "reranker_score" in source:
                        st.caption(
                            "Reranker score: "
                            f"{source['reranker_score']}"
                        )

        except Exception as exc:
            st.error(
                f"Unable to answer the question: {exc}"
            )


# ============================================================
# 4. Inspect the S3 vector index
# ============================================================

st.divider()
st.header("4. Inspect indexed chunks")

if st.button("Load indexed chunk summary"):
    try:
        with st.spinner("Loading vector chunks from S3..."):
            chunks = load_s3_vector_chunks()

        st.metric("Chunks loaded", len(chunks))

        if chunks:
            summary = []

            for chunk in chunks:
                summary.append({
                    "Document": chunk.get(
                        "document_name",
                        "Unknown",
                    ),
                    "Page": chunk.get(
                        "page",
                        "Unknown",
                    ),
                    "Chunk ID": chunk.get(
                        "chunk_id",
                        "Unknown",
                    ),
                    "Embedding dimensions": len(
                        chunk.get("embedding", [])
                    ),
                    "Text preview": chunk.get(
                        "text",
                        "",
                    )[:200],
                })

            st.dataframe(
                pd.DataFrame(summary),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No valid embedded chunks were found in S3."
            )

    except Exception as exc:
        st.error(
            f"Could not load indexed chunks: {exc}"
        )


# ============================================================
# 5. Legacy Hyderabad bucket
# ============================================================

with st.expander("Legacy Hyderabad S3 documents"):
    st.write(
        "This section lists PDFs in the older S3 bucket. "
        "It is separate from the current Mumbai processing pipeline."
    )

    if st.button("Refresh Hyderabad S3 documents"):
        try:
            old_files = list_old_bucket_pdfs()
            show_object_table(
                old_files,
                "No PDFs found under raw/ in the old bucket.",
            )
        except Exception as exc:
            st.error(
                f"Could not retrieve Hyderabad S3 documents: {exc}"
            )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Nullfinity | Amazon S3 | AWS Lambda OCR | "
    "Hugging Face Embeddings | S3 Vector Index | "
    "Hybrid RAG | BM25 | Reciprocal Rank Fusion | "
    "Cross-Encoder Reranking | Groq"
)
