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
import pandas as pd
import json
import re

# Constants
KEYS: List[str] = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
MODEL_NAME: str = "gemini-1.5-pro"
UPLOAD_DIR: str = "uploads"

# Configure genai
genai.configure(api_key=random.choice(KEYS))

# Hide Streamlit elements
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_gemini_response(user_input: str, prompt: str, context: str, history: List[Dict[str, str]]) -> str:
    generation_config = {
        "temperature": 0.2,
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

@st.cache_data
def process_uploaded_file(file_path: str) -> str:
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() in ['.csv', '.xlsx', '.xls']:
        if file_extension.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df.to_string()
    elif file_extension.lower() == '.pdf':
        elements = partition(filename=file_path)
        return "\n\n".join(str(el) for el in elements)
    else:
        return "Unsupported file type"

@st.cache_data
def clean_json_string(json_string: str) -> List[Dict[str, str]]:
    json_string = json_string.strip()
    json_match = re.search(r'\[.*\]', json_string, re.DOTALL)
    if json_match:
        json_string = json_match.group()
    else:
        return []
    
    json_string = json_string.replace("'", '"')
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        items = re.findall(r'\{.*?\}', json_string)
        result = []
        for item in items:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return result

@st.cache_data
def generate_prompts(file_content: str) -> List[Dict[str, str]]:
    prompt = f"""
    Based on the following file content, generate 5 relevant prompts that a user might want to ask about the data.
    Format the prompts as a JSON list of dictionaries, where each dictionary has a 'prompt' key and a 'description' key.

    File content:
    {file_content[:1000]}

    THE OUTPUT MUST BE A VALID JSON LIST OF DICTIONARIES. EXAMPLE FORMAT:
    [
        {{"prompt": "Summarize the main points", "description": "Get an overview of the key information"}},
        {{"prompt": "Analyze trends in the data", "description": "Identify patterns and trends in the dataset"}}
    ]
    """
    
    response = get_gemini_response("", prompt, "", [])
    cleaned_response = clean_json_string(response)
    return cleaned_response

def main():
    st.title("File Processing Agent")
    
    if 'file_content' not in st.session_state:
        st.session_state.file_content = None
    if 'prompts' not in st.session_state:
        st.session_state.prompts = None
    if 'responses' not in st.session_state:
        st.session_state.responses = {}

    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'pdf'])
    
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.file_content = process_uploaded_file(file_path)
        st.text_area("File Content Preview", st.session_state.file_content[:1000], height=200)
        
        if st.session_state.prompts is None:
            st.session_state.prompts = generate_prompts(st.session_state.file_content)
        
        st.subheader("Suggested Prompts")
        if st.session_state.prompts:
            for i, prompt in enumerate(st.session_state.prompts):
                if st.button(prompt['description'], key=f"prompt_{i}"):
                    if i not in st.session_state.responses:
                        response = get_gemini_response(prompt['prompt'], "", st.session_state.file_content, [])
                        st.session_state.responses[i] = response
                    st.write(st.session_state.responses[i])
        else:
            st.warning("No prompts were generated. Please try again or check the file content.")

if __name__ == "__main__":
    main()