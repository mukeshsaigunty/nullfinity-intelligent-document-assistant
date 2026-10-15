
FROM public.ecr.aws/lambda/python:3.12

# Install PDF utilities
RUN dnf install -y poppler-utils \
    && dnf clean all

# Copy the Tesseract executable
COPY aws-lambda-tesseract-layer/ready-to-use/amazonlinux-2023/bin/tesseract /usr/local/bin/tesseract

# Copy Tesseract shared libraries
COPY aws-lambda-tesseract-layer/ready-to-use/amazonlinux-2023/lib/ /usr/local/lib/

# Copy OCR language data
COPY aws-lambda-tesseract-layer/ready-to-use/amazonlinux-2023/tesseract/share/tessdata/ /usr/share/tessdata/

# Configure Tesseract
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV TESSDATA_PREFIX=/usr/share/tessdata/

RUN chmod +x /usr/local/bin/tesseract

# Install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy Lambda function
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_function.lambda_handler"]
