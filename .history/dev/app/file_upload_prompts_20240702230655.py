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
MODEL_NAME: str = "gemini-1.5-pro"  # Changed to pro model for better performance
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
        "temperature": 0.2,  # Lowered temperature for more focused outputs
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


import re
def clean_json_string(json_string: str) -> str:
    # Remove any leading/trailing whitespace and newlines
    json_string = json_string.strip()
    
    # Remove any text before the first '[' and after the last ']'
    json_string = re.search(r'\[.*\]', json_string, re.DOTALL)
    if json_string:
        json_string = json_string.group()
    else:
        return ""
    
    # Replace any single quotes with double quotes
    json_string = json_string.replace("'", '"')
    
    return json_string
def generate_prompts(file_content: str) -> List[Dict[str, str]]:
    prompt = f"""
    Based on the following file content, generate 5 relevant prompts that a user might want to ask about the data.
    Format the prompts as a JSON list of dictionaries, where each dictionary has a 'prompt' key and a 'description' key.

    File content:
    {file_content[:1000]}  # Limit to first 1000 characters to avoid overwhelming the model

    THE OUTPUT MUST BE A VALID JSON LIST OF DICTIONARIES. EXAMPLE FORMAT:
    [
        {{"prompt": "Summarize the main points", "description": "Get an overview of the key information"}},
        {{"prompt": "Analyze trends in the data", "description": "Identify patterns and trends in the dataset"}}
    ]
    """
    
    response = get_gemini_response("", prompt, "", [])
    cleaned_response = clean_json_string(response)
    
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON. Error: {str(e)}")
        st.error(f"Cleaned response: {cleaned_response}")
        return []


def main():
    st.title("File Processing Agent")
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'pdf'])
    
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure the upload directory exists
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_content = process_uploaded_file(file_path)
        st.text_area("File Content Preview", file_content[:1000], height=200)
        
        prompts = generate_prompts(file_content)
        
        st.subheader("Suggested Prompts")
        if prompts:
            for prompt in prompts:
                if st.button(prompt['description']):
                    response = get_gemini_response(prompt['prompt'], "", file_content, [])
                    st.write(response)
        else:
            st.warning("No prompts were generated. Please try again or check the file content.")

if __name__ == "__main__":
    main()
