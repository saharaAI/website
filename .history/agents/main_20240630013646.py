import hashlib
import os
import random
from typing import List, Dict

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

def get_gemini_response(user_input: str, prompt: str, context: str = "", history: List[Dict[str, str]] = []) -> str:
    """Generate a response using the Gemini model."""
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

    # Convert history to the format expected by Gemini
    formatted_history = []
    for message in history:
        if message["role"] == "user":
            formatted_history.append({"role": "user", "parts": [message["content"]]})
        elif message["role"] == "assistant":
            formatted_history.append({"role": "model", "parts": [message["content"]]})

    chat_session = model.start_chat(history=formatted_history)

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

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    
    if 'doc_content' not in st.session_state:
        st.session_state.doc_content = ""

    st.sidebar.title("Configuration")
    prompt = st.sidebar.text_area('Invite de système', value='Agir comme un expert en analyse de documents.')
    uploaded_file = st.sidebar.file_uploader('Télécharger un fichier PDF', type=['pdf'])

    if uploaded_file and not st.session_state.file_processed:
        with st.spinner('Traitement du fichier en cours...'):
            file_path = save_uploaded_file(uploaded_file)
            st.session_state.doc_content = process_uploaded_file(file_path)
            st.session_state.file_processed = True
        st.sidebar.success("Fichier téléchargé et traité avec succès!")

    st.title('📄 Analyse de PDF avec Gemini')

    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # User input
    user_input = st.chat_input('Posez votre question sur le document')
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner('Génération de la réponse...'):
                prompt_with_doc = f'{prompt}\n{st.session_state.doc_content}'
                response = get_gemini_response(user_input, prompt_with_doc, context=st.session_state.doc_content, history=st.session_state.chat_history)
            st.write(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Clear chat button
    if st.button('Effacer la conversation'):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == '__main__':
    main()