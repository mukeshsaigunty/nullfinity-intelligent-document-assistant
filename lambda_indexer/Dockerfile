
FROM public.ecr.aws/lambda/python:3.12

ENV HF_HOME=/opt/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/opt/huggingface
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10


# Install CPU-only PyTorch to avoid CUDA/NVIDIA dependencies
RUN python -m pip install --no-cache-dir \
    --timeout 120 --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.6.0"

# Install pinned embedding dependencies
RUN python -m pip install --no-cache-dir \
    --timeout 120 --retries 10 \
    --index-url https://pypi.org/simple \
    "numpy==1.26.4" \
    "transformers==4.48.3" \
    "sentence-transformers==3.4.1"

# Download the embedding model during image build
RUN python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/opt/huggingface'); print('Embedding model loaded:', m.get_sentence_embedding_dimension())"

COPY lambda_function.py ${LAMBDA_TASK_ROOT}/lambda_function.py

CMD ["lambda_function.lambda_handler"]
