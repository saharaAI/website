from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os

import random

# --- Configuration ---

# API Keys (replace with your actual keys)
KEYs = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU" ,"AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
KEY = random.choice(KEYs)
genai.configure(api_key=KEY)
# Initialize FastAPI app
app = FastAPI()

class Prompt(BaseModel):
    prompt: str

def get_response(prompt):
    """Handles Gemini API calls."""
    generation_config = {
        "temperature": 0.5,  # Adjust for creativity (0.0 - deterministic, 1.0 - very creative)
        "top_p": 0.95,      # Controls the diversity of the generated text
        "top_k": 40,         # Limits the vocabulary from which Gemini can sample tokens
        "max_output_tokens": 2048, # Adjust based on the expected response length 
    }

    model = genai.GenerativeModel(
        model_name="gemini-pro",  # Or use a different Gemini model
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [prompt]}],
    )

    response = chat_session.send_message(prompt)
    return response.text

@app.post("/generate")
def generate_text(prompt: Prompt):
    try:
        response = get_response(prompt.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
