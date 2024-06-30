import hashlib
import os
import random
from typing import List, Dict
import io
import streamlit as st
import google.generativeai as genai
from unstructured.partition.auto import partition
import markdown
import pyperclip

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

def copy_to_clipboard(text: str):
    pyperclip.copy(text)

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

    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            content = message.get("parts", [message.get("content", "")])[0]
            st.write(content)
            if message["role"] == "model":
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    if st.button(f"📋 Copier", key=f"copy_{i}"):
                        copy_to_clipboard(content)
                        st.success("Copié dans le presse-papiers!")
                with col2:
                    md_buffer = export_chat([message], "md")
                    st.download_button(
                        label="📥 MD",
                        data=md_buffer.getvalue(),
                        file_name=f"response_{i}.md",
                        mime="text/markdown",
                        key=f"md_{i}"
                    )
                with col3:
                    html_buffer = export_chat([message], "html")
                    st.download_button(
                        label="📥 HTML",
                        data=html_buffer.getvalue(),
                        file_name=f"response_{i}.html",
                        mime="text/html",
                        key=f"html_{i}"
                    )

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
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button(f"📋 Copier", key=f"copy_{len(st.session_state.chat_history)}"):
                    copy_to_clipboard(response)
                    st.success("Copié dans le presse-papiers!")
            with col2:
                md_buffer = export_chat([{"role": "model", "parts": [response]}], "md")
                st.download_button(
                    label="📥 MD",
                    data=md_buffer.getvalue(),
                    file_name=f"response_{len(st.session_state.chat_history)}.md",
                    mime="text/markdown",
                    key=f"md_{len(st.session_state.chat_history)}"
                )
            with col3:
                html_buffer = export_chat([{"role": "model", "parts": [response]}], "html")
                st.download_button(
                    label="📥 HTML",
                    data=html_buffer.getvalue(),
                    file_name=f"response_{len(st.session_state.chat_history)}.html",
                    mime="text/html",
                    key=f"html_{len(st.session_state.chat_history)}"
                )
        
        st.session_state.chat_history.append({"role": "model", "parts": [response]})