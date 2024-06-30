import hashlib
import os
import random
from typing import List

import streamlit as st
import google.generativeai as genai
from unstructured.partition.auto import partition

# Constants
KEYS: List[str] = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
MODEL_NAME: str = "gemini-1.5-flash"
UPLOAD_DIR: str = "uploads"

# Configure genai
genai.configure(api_key=random.choice(KEYS))

def process_uploaded_file(file_path: str) -> str:
    """Process the uploaded file and extract its content."""
    elements = partition(filename=file_path)
    file_content = "\n\n".join(str(el) for el in elements)
    return file_content

def get_response(user_input: str, prompt: str, context: str = "") -> str:
    """Generate a response using the AI model."""
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [prompt]}],
    )

    if context:
        chat_session.send_message(context)
    
    response = chat_session.send_message(user_input)
    return response.text

def save_uploaded_file(uploaded_file) -> str:
    """Save the uploaded file and return its path."""
    file_bytes = uploaded_file.read()
    prefix = hashlib.md5(file_bytes).hexdigest()
    filename = f'{prefix}.pdf'
    file_path = os.path.join(UPLOAD_DIR, filename)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(file_path, 'wb') as tmpfile:
        tmpfile.write(file_bytes)
    
    return file_path

def main():
    st.set_page_config(
        layout='wide',
        page_title='Analyse de PDF',
        page_icon='📄'
    )
    st.header('📄 Analyse de PDF avec Lagent')

    prompt = st.sidebar.text_area('Invite de système', value='Agir comme un expert.')
    uploaded_file = st.sidebar.file_uploader('Télécharger un fichier PDF', type=['pdf'])

    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        doc_content = process_uploaded_file(file_path)
        prompt_with_doc = f'{prompt}\n{doc_content}'

        st.success("Fichier téléchargé et traité avec succès!")

        user_input = st.chat_input('Posez votre question sur le document')
        if user_input:
            with st.spinner('Génération de la réponse...'):
                response = get_response(user_input, prompt_with_doc, context=doc_content)
            
            st.markdown(f"**Utilisateur:** {user_input}")
            st.markdown(f"**Assistant:** {response}")

if __name__ == '__main__':
    main()