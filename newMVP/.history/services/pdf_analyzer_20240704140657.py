import hashlib
import os
from unstructured.partition.auto import partition
from config import UPLOAD_DIR

def process_uploaded_file(file_path: str) -> str:
    elements = partition(filename=file_path)
    return "\n\n".join(str(el) for el in elements)

def save_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.read()
    prefix = hashlib.md5(file_bytes).hexdigest()
    filename = f'{prefix}.pdf'
    file_path = os.path.join(UPLOAD_DIR, filename)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(file_path, 'wb') as tmpfile:
        tmpfile.write(file_bytes)
    
    return file_path