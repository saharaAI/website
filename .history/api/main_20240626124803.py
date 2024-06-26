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

@app.get("/")
def read_root():
    return {"Home": "Sahara Analytics - MVP"}

def get_prompt(number: int, context: str):
    d = {
        1: "Act as a financial expert and provide a detailed analysis of the following financial statement: " + context,
        2: "Act as a financial expert and provide a detailed analysis of the following credit report: " + context,
    }
    return d[number]
def get_response(user_input):   
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
        history=[{"role": "user", "parts": [user_input]}],
    )
    return chat_session.send_message(user_input).text

@app.post("/generate")
def generate_text(prompt: Prompt):
    try:
        response = get_response(prompt.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

