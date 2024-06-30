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
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stTextInput>div>div>input {
        background-color: white;
        color: #333;
        border: 1px solid #bdc3c7;
        border-radius: 4px;
    }
    .user-message {
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .assistant-message {
        background-color: #d5e5f2;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

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
    user_input = st.chat_input('Votre question :', key='user_input')
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