import copy
import hashlib
import json
import os
import random

import streamlit as st

# Importez le modèle Gemini
import google.generativeai as genai

KEYs = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
KEY = random.choice(KEYs)
genai.configure(api_key=KEY)

from unstructured.partition.auto import partition

def process_uploaded_file(file_path):
    elements = partition(filename=file_path)
    file_content = "\n\n".join([str(el) for el in elements] + [""])
    print(file_content)
    return file_content

# Configuration pour le modèle Gemini
def get_response(user_input, prompt):   
    generation_config = {
        "temperature": 0.5,  
        "top_p": 0.95,     
        "top_k": 40,        
        "max_output_tokens": 2048,  
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [prompt]}],
    )

    response = chat_session.send_message(user_input)
    return response.text

class SessionState:

    def init_state(self):
        """Initialisez les variables d'état de session."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['model_selected'] = None
        st.session_state['history'] = []

    def clear_state(self):
        """Réinitialisez l'état de session existant."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['model_selected'] = None
        st.session_state['file'] = set()

class StreamlitUI:

    def __init__(self, session_state: SessionState):
        self.init_streamlit()
        self.session_state = session_state

    def init_streamlit(self):
        """Initialisez les paramètres de l'interface utilisateur de Streamlit."""
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        st.header(':robot_face: :blue[Lagent] Démo Web', divider='rainbow')
        st.sidebar.title('Contrôle du modèle')
        st.session_state['file'] = set()
        st.session_state['model_path'] = None

    def setup_sidebar(self):
        """Configurez la barre latérale pour la sélection des modèles et des plugins."""
        model_name = st.sidebar.text_input('Nom du modèle :', value='gemini-1.5-flash')
        prompt = st.sidebar.text_area('Invite de système', value='Acter comme expert.')

        if model_name != st.session_state['model_selected']:
            st.session_state['model_selected'] = model_name
            self.session_state.clear_state()
            if 'chatbot' in st.session_state:
                del st.session_state['chatbot']

        if st.sidebar.button('Effacer la conversation', key='clear'):
            self.session_state.clear_state()
        uploaded_file = st.sidebar.file_uploader('Télécharger un fichier')

        return model_name, prompt, uploaded_file

    def render_user(self, prompt: str):
        with st.chat_message('user'):
            st.markdown(prompt)

    def render_assistant(self, response):
        with st.chat_message('assistant'):
            st.markdown(response)

def main():
    if 'ui' not in st.session_state:
        session_state = SessionState()
        session_state.init_state()
        st.session_state['ui'] = StreamlitUI(session_state)
    else:
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        st.header(':robot_face: :blue[Lagent] Démo Web', divider='rainbow')

    _, prompt, uploaded_file = st.session_state['ui'].setup_sidebar()

    for prompt_text, response in zip(st.session_state['user'], st.session_state['assistant']):
        st.session_state['ui'].render_user(prompt_text)
        st.session_state['ui'].render_assistant(response)

    if user_input := st.chat_input(''):
        with st.container():
            st.session_state['ui'].render_user(user_input)
        st.session_state['user'].append(user_input)
        response = get_response(user_input, prompt)
        st.session_state['assistant'].append(response)
        st.session_state['ui'].render_assistant(response)

        # Ajoutez le téléchargeur de fichiers à la barre latérale
        if uploaded_file and uploaded_file.name not in st.session_state['file']:
            st.session_state['file'].add(uploaded_file.name)
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            if 'image' in file_type:
                st.image(file_bytes, caption='Image téléchargée')
            elif 'video' in file_type:
                st.video(file_bytes, caption='Vidéo téléchargée')
            elif 'audio' in file_type:
                st.audio(file_bytes, caption='Audio téléchargé')

            postfix = uploaded_file.name.split('.')[-1]
            prefix = hashlib.md5(file_bytes).hexdigest()
            filename = f'{prefix}.{postfix}'
            file_path = os.path.join(root_dir, filename)
            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(file_bytes)
            file_size = os.stat(file_path).st_size / 1024 / 1024
            file_size = f'{round(file_size, 2)} MB'
            user_input = [
                dict(role='user', content=user_input),
                dict(
                    role='user',
                    content=json.dumps(dict(path=file_path, size=file_size)),
                    name='file')
            ]

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_dir, 'tmp_dir')
    os.makedirs(root_dir, exist_ok=True)
    main()
