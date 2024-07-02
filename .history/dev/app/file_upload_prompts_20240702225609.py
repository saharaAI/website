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

# Constants
KEYS: List[str] = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
MODEL_NAME: str = "gemini-1.5-flash"
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

def generate_prompts(file_content: str) -> List[Dict[str, str]]:
    prompt = f"""
    Based on the following file content, generate 5 relevant prompts that a user might want to ask about the data. 
    Format the prompts as a JSON list of dictionaries, where each dictionary has a 'prompt' key and a 'description' key.
    
    File content:
    {file_content[:1000]}  # Limit to first 1000 characters to avoid overwhelming the model
    
    Example output format:
    [
        {{"prompt": "Summarize the main points", "description": "Get an overview of the key information"}},
        {{"prompt": "Analyze trends in the data", "description": "Identify patterns and trends in the dataset"}}
    ]
    """
    
    response = get_gemini_response("", prompt, "", [])
    return json.loads(response)

def main():
    st.title("File Processing Agent")
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'pdf'])
    
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_content = process_uploaded_file(file_path)
        st.text_area("File Content Preview", file_content[:1000], height=200)
        
        prompts = generate_prompts(file_content)
        
        st.subheader("Suggested Prompts")
        for prompt in prompts:
            if st.button(prompt['description']):
                response = get_gemini_response(prompt['prompt'], "", file_content, [])
                st.write(response)

if __name__ == "__main__":
    main()