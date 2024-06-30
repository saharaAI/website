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

st.set_page_config(layout='wide', page_title='Analyse de Document', page_icon='📄')

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
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

    # Initialisation des variables de session
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    if 'doc_content' not in st.session_state:
        st.session_state.doc_content = ""

    # Interface utilisateur personnalisée
    st.markdown("<h1 style='text-align: center;'>📄 Analyse de Document</h1>", unsafe_allow_html=True)

    # Sidebar personnalisée
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<h3 style='text-align: center;'>Configuration</h3>", unsafe_allow_html=True)
        prompt = st.text_area('Instructions système', value='Agir comme un expert en analyse de documents.')
        uploaded_file = st.file_uploader('Télécharger un fichier', type=['pdf'], key='file_uploader')

    # Traitement du fichier
    if uploaded_file and not st.session_state.file_processed:
        with st.spinner('Traitement du fichier en cours...'):
            file_path = save_uploaded_file(uploaded_file)
            st.session_state.doc_content = process_uploaded_file(file_path)
            st.session_state.file_processed = True
        st.success("Fichier traité avec succès!")

    # Affichage de l'historique du chat
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>Vous :</strong> {message['parts'][0]}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>Assistant :</strong> {message['parts'][0]}</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button(f"📋 Copier", key=f"copy_{i}"):
                    copy_to_clipboard(message['parts'][0])
                    st.success("Copié!")
            with col2:
                md_buffer = export_chat([message], "md")
                st.download_button("📥 MD", data=md_buffer.getvalue(), file_name=f"response_{i}.md", mime="text/markdown", key=f"md_{i}")
            with col3:
                html_buffer = export_chat([message], "html")
                st.download_button("📥 HTML", data=html_buffer.getvalue(), file_name=f"response_{i}.html", mime="text/html", key=f"html_{i}")

    # Zone de saisie utilisateur
    user_input = st.text_input('Votre question :', key='user_input')
    if st.button('Envoyer'):
        if user_input:
            st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
            with st.spinner('Génération de la réponse...'):
                response = get_gemini_response(user_input, prompt, st.session_state.doc_content, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "model", "parts": [response]})
            st.experimental_rerun()

    # Boutons d'exportation et de réinitialisation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Exporter en MD'):
            md_buffer = export_chat(st.session_state.chat_history, "md")
            st.download_button("Télécharger MD", data=md_buffer.getvalue(), file_name="chat_history.md", mime="text/markdown")
    with col2:
        if st.button('Exporter en HTML'):
            html_buffer = export_chat(st.session_state.chat_history, "html")
            st.download_button("Télécharger HTML", data=html_buffer.getvalue(), file_name="chat_history.html", mime="text/html")
    with col3:
        if st.button('Nouvelle conversation'):
            st.session_state.chat_history = []
            st.session_state.file_processed = False
            st.session_state.doc_content = ""
            st.experimental_rerun()

if __name__ == '__main__':
    main()