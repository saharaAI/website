import hashlib
import os
import random

import streamlit as st

import google.generativeai as genai
from unstructured.partition.auto import partition

KEYs = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
KEY = random.choice(KEYs)
genai.configure(api_key=KEY)

def process_uploaded_file(file_path):
    elements = partition(filename=file_path)
    file_content = "\n\n".join([str(el) for el in elements] + [""])
    print(file_content)
    return file_content

def get_response(user_input, prompt, context=""):  
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

    if context:
        chat_session.send_message(context)
    
    response = chat_session.send_message(user_input)
    return response.text

def main():
    st.set_page_config(
        layout='wide',
        page_title='Analyse de PDF',
        page_icon='📄'
    )
    st.header('📄 Analyse de PDF avec Lagent')

    prompt = st.sidebar.text_area('Invite de système', value='Acter comme expert.')
    uploaded_file = st.sidebar.file_uploader('Télécharger un fichier PDF', type=['pdf'])

    if uploaded_file:
        file_bytes = uploaded_file.read()
        prefix = hashlib.md5(file_bytes).hexdigest()
        filename = f'{prefix}.pdf'
        file_path = os.path.join('uploads', filename)

        os.makedirs('uploads', exist_ok=True)
        with open(file_path, 'wb') as tmpfile:
            tmpfile.write(file_bytes)
        
        doc_content = process_uploaded_file(file_path)
        prompt_with_doc = f'{prompt}\n{doc_content}'

        if user_input := st.chat_input(''):
            response = get_response(user_input, prompt_with_doc, context=doc_content)
            st.markdown(f"**Utilisateur:** {user_input}")
            st.markdown(f"**Assistant:** {response}")

if __name__ == '__main__':
    main()
