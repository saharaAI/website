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
from utils import *
# Constants
KEYS: List[str] = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
MODEL_NAME: str = "gemini-1.5-flash"
UPLOAD_DIR: str = "uploads"


# Configure genai
genai.configure(api_key=random.choice(KEYS))

# Configuration de la page Streamlit
st.set_page_config(layout='wide', page_title='Analyse de Document', page_icon='📄')

# Style CSS personnalisé
st.markdown("""
<style>
    body {
        color: #333;
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #0066cc;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0052a3;
    }
    .stTextInput > div > div > input {
        background-color: white;
        color: #333;
        border: 1px solid #ced4da;
        border-radius: 4px;
    }
    .user-message {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .assistant-message {
        background-color: #e6f3ff;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f5;
    }
</style>
""", unsafe_allow_html=True)

# Cacher les éléments de Streamlit
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Fonctions existantes (process_uploaded_file, get_gemini_response, save_uploaded_file, export_chat, copy_to_clipboard)
# ...

def main():
    # Initialisation des variables de session
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    if 'doc_content' not in st.session_state:
        st.session_state.doc_content = ""

    st.title('📄 Analyse de Document')

    # Sidebar
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
            st.markdown(f"<div class='user-message'><strong>Vous :</strong> {message['parts'][0]}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><strong>Assistant :</strong> {message['parts'][0]}</div>", unsafe_allow_html=True)
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
    user_input = st.chat_input('Votre question :', key='user_input')
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