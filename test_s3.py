
from src.s3_service import list_s3_pdfs

try:
    files = list_s3_pdfs()

    print("S3 connection successful!")
    print(f"PDF files found: {len(files)}")

    for file in files:
        print(f"File: {file['key']}")
        print(f"Size: {file['size_bytes']} bytes")

except Exception as exc:
    print(f"S3 test failed: {exc}")
