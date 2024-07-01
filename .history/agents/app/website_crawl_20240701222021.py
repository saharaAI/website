import streamlit as st
import trafilatura
from trafilatura.spider import focused_crawler

from urllib.parse import urlparse
import google.generativeai as genai
from typing import List, Dict
import os
import hashlib
import random
# Constants
KEYS= ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
MODEL_NAME = "gemini-1.5-flash" 
UPLOAD_DIR = "uploads"

# Configure genai
genai.configure(api_key=random.choice(KEYS))
class WebScraper:
    def __init__(self, url):
        self.url = url
        self.extracted_text = ""
        self.metadata = {}

    def scrape(self):
        try:
            downloaded = trafilatura.fetch_url(self.url)
            self.extracted_text = trafilatura.extract(downloaded)
        except Exception as e:
            st.error(f"An error occurred during scraping: {e}")

class GeminiApp:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.generation_config = {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

    def generate_response(self, user_input: str, context: str, history: List[Dict[str, str]]) -> str:
        model = genai.GenerativeModel(model_name=self.model_name, generation_config=self.generation_config)
        
        formatted_history = [
            {"role": "user" if msg["role"] == "user" else "model", "parts": [msg.get("parts", [msg.get("content", "")])[0]]}
            for msg in history
        ]
        
        chat_session = model.start_chat(history=formatted_history)
        full_prompt = f"You are a helpful and informative chatbot. Here's the context from a web page:\n\n{context}\n\nUser question: {user_input}\n\nAnswer:"
        return chat_session.send_message(full_prompt).text

def save_extracted_text(text, filename):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path


def main():
    st.title("Web Scraper & Gemini App")

    # Input URL and Prompt
    url = st.text_input("Enter the URL of the website:", value = 'https://www.bcm.mr/')
    prompt = st.text_area("Enter your prompt (optional):", "Summarize the key points from the extracted text.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Scrape and Analyze"):
        if url:
            # Scrape the website
            scraper = WebScraper(url)
            scraper.scrape()

            # Display extracted content and metadata
            st.subheader("Extracted Text:")
            st.text_area(" ", scraper.extracted_text, height=200)


            # Generate response using Gemini
            gemini = GeminiApp()
            response = gemini.generate_response(prompt, scraper.extracted_text, st.session_state.chat_history)
            st.subheader("Gemini Response:")
            st.write(response)

            # Update chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "model", "content": response})

        else:
            st.warning("Please enter a valid URL.")

if __name__ == "__main__":
    main()