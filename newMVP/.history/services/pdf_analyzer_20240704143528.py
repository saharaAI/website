import hashlib
import os
from unstructured.partition.auto import partition
from config import UPLOAD_DIR

import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

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