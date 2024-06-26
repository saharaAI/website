import google.generativeai as genai
import random
from dotenv import load_dotenv
import os 

load_dotenv()

# --- Gemini API Configuration ---
# Safely load your API key from environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 

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
    return response