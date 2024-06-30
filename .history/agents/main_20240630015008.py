import hashlib
import os
import random
from typing import List, Dict
import io

import streamlit as st
import google.generativeai as genai
from unstructured.partition.auto import partition
import markdown

# Constants
KEYS: List[str] = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
MODEL_NAME: str = "gemini-1.5-flash"
UPLOAD_DIR: str = "uploads"

# Configure genai
genai.configure(api_key=random.choice(KEYS))

@st.cache_data
def process_uploaded_file(file_path: str) -> str:
    elements = partition(filename=file_path)
    return "\n\n".join(str(el) for el in elements)

def get_gemini_response(user_input: str, prompt: str, context: str, history: List[Dict[str, str]]) -> str:
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }

    model = genai.GenerativeModel(model_name=MODEL_NAME, generation_config=generation_config)

    formatted_history = [
        {"role": "user" if msg["role"] == "user" else "model", "parts": [msg.get("parts", [msg.get("content", "")])[0]]}
        for msg in history
    ]

    chat_session = model.start_chat(history=formatted_history)
    full_prompt = f"{prompt}\n\nDocument content:\n{context}\n\nUser question: {user_input}\n\nAnswer:"
    
    return chat_session.send_message(full_prompt).text

def save_uploaded_file(uploaded_file) -> str:
    file_bytes = uploaded_file.read()
    prefix = hashlib.md5(file_bytes).hexdigest()
    filename = f'{prefix}.pdf'
    file_path = os.path.join(UPLOAD_DIR, filename)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(file_path, 'wb') as tmpfile:
        tmpfile.write(file_bytes)
    
    return file_path

def export_chat(chat_history: List[Dict[str, str]], format: str) -> io.BytesIO:
    md_content = "# Chat History\n\n"
    for message in chat_history:
        role = message["role"]
        content = message.get("parts", [message.get("content", "")])[0]
        md_content += f"## {role.capitalize()}\n\n{content}\n\n"

    if format == "md":
        return io.BytesIO(md_content.encode())
    elif format == "html":
        html_content = markdown.markdown(md_content)
        return io.BytesIO(html_content.encode())

def main():
    st.set_page_config(layout='wide', page_title='Analyse de PDF', page_icon='📄')

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

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message.get("parts", [message.get("content", "")])[0])

    user_input = st.chat_input('Posez votre question sur le document')
    if user_input:
        st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner('Génération de la réponse...'):
                response = get_gemini_response(
                    user_input=user_input,
                    prompt=prompt,
                    context=st.session_state.doc_content,
                    history=st.session_state.chat_history
                )
            st.write(response)
        
        st.session_state.chat_history.append({"role": "model", "parts": [response]})

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Exporter en MD'):
            md_buffer = export_chat(st.session_state.chat_history, "md")
            st.download_button(
                label="Télécharger MD",
                data=md_buffer.getvalue(),
                file_name="chat_history.md",
                mime="text/markdown"
            )
    with col2:
        if st.button('Exporter en HTML'):
            html_buffer = export_chat(st.session_state.chat_history, "html")
            st.download_button(
                label="Télécharger HTML",
                data=html_buffer.getvalue(),
                file_name="chat_history.html",
                mime="text/html"
            )

    if st.button('Effacer la conversation'):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == '__main__':
    main()