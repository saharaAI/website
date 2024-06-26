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

P = {
        1: "Act as a financial expert and provide a detailed analysis of the following financial statement: " ,
        2: "Act as a financial expert and provide a detailed analysis of the following credit report: " ,
        3: "Act like expert in python and provide a detailed analysis : "
    }

@app.get("/")
def read_root():
    return {"Home": "Sahara Analytics - MVP"}

def get_prompt(number: int, context: str):
    d = P.copy()
    if context:
        d[number] = d[number] + f" \n CONTEXT: {context}"
    return d[number]
def get_response(user_input,prompt):   
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

# show prompt list
@app.get("/prompts")
def get_prompts():  
    return {"prompts": P}
@app.post("/generate")
def generate(message: str, number: int, context: str):
    prompt = get_prompt(number, context)
    response = get_response(message, prompt)
    return {"response": response}
