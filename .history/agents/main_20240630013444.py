import hashlib
import os
import random
from typing import List, Dict

import streamlit as st
import google.generativeai as genai
from unstructured.partition.auto import partition
import autollm

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

    chat_session = model.start_chat(history=history)

    if context:
        chat_session.send_message(context)
    
    response = chat_session.send_message(user_input)
    return response.text

def get_autollm_response(user_input: str, context: str, role: str, target: str) -> str:
    """Generate a response using AutoLLM with role-playing and target."""
    system_prompt = f"Tu es un {role} expert. Ton objectif est de {target}. Utilise le contexte suivant pour répondre à la question de l'utilisateur."
    
    full_prompt = f"{system_prompt}\n\nContexte:\n{context}\n\nQuestion de l'utilisateur: {user_input}\n\nRéponse:"
    
    response = autollm.generate(full_prompt)
    return response

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

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    model_choice = st.sidebar.selectbox('Choisir le modèle', ['Gemini', 'AutoLLM'])
    
    if model_choice == 'Gemini':
        prompt = st.sidebar.text_area('Invite de système', value='Agir comme un expert.')
    else:
        role = st.sidebar.text_input('Rôle de l\'IA', value='analyste financier')
        target = st.sidebar.text_input('Objectif de l\'IA', value='fournir des analyses financières détaillées')
    
    uploaded_file = st.sidebar.file_uploader('Télécharger un fichier PDF', type=['pdf'])

    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        doc_content = process_uploaded_file(file_path)

        st.success("Fichier téléchargé et traité avec succès!")

        # Display chat history
        for message in st.session_state.chat_history:
            st.markdown(f"**{message['role']}:** {message['content']}")

        user_input = st.chat_input('Posez votre question sur le document')
        if user_input:
            st.session_state.chat_history.append({"role": "Utilisateur", "content": user_input})
            
            with st.spinner('Génération de la réponse...'):
                if model_choice == 'Gemini':
                    prompt_with_doc = f'{prompt}\n{doc_content}'
                    response = get_gemini_response(user_input, prompt_with_doc, context=doc_content, history=st.session_state.chat_history)
                else:
                    response = get_autollm_response(user_input, doc_content, role, target)
            
            st.session_state.chat_history.append({"role": "Assistant", "content": response})
            st.markdown(f"**Assistant:** {response}")

if __name__ == '__main__':
    main()