from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import random
import json

# --- Configuration ---
# You can easily manage API keys and configurations here.
class Config:
    """Configuration class to store API keys and other settings."""
    
    API_KEYS =  ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU" ,"AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
    MODEL_NAME = "gemini-1.5-flash"
    GENERATION_CONFIG = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }

# --- Models ---

class PromptRequest(BaseModel):
    """Request model for prompt selection."""
    number: int
    context: Optional[str] = None  # Optional context for the prompt

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    message: str
    prompt: PromptRequest

# --- Credit Risk Analysis Engine ---

class CreditRiskAnalyzer:
    """
    A class to encapsulate the credit risk analysis logic using Google Gemini.
    """

    def __init__(self, config: Config):
        """Initializes the analyzer with the given configuration."""
        self.config = config
        genai.configure(api_key=random.choice(self.config.API_KEYS))
        self.model = genai.GenerativeModel(
            model_name=self.config.MODEL_NAME, 
            generation_config=self.config.GENERATION_CONFIG
        )
        self.prompts = json.load(open("prompts.json", "r"))

    def get_prompt(self, prompt_request: PromptRequest) -> str:
        """Retrieves the selected prompt with optional context."""
        if prompt_request.number not in self.prompts:
            raise HTTPException(status_code=400, detail="Invalid prompt number")

        prompt = self.prompts[prompt_request.number]
        if prompt_request.context:
            prompt += f"\nCONTEXT: {prompt_request.context}"
        return prompt

    def analyze(self, message: str, prompt: str) -> str:
        """Performs credit risk analysis using the Gemini model."""
        chat_session = self.model.start_chat(
            history=[{"role": "user", "parts": [prompt]}]
        )
        response = chat_session.send_message(message)
        return response.text

# --- FastAPI app ---

app = FastAPI(
    title="Sahara Analytics API",
    description="An API for credit risk analysis using Google Gemini.",
    version="0.1.0"
)

analyzer = CreditRiskAnalyzer(Config)  # Initialize the analyzer

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to Sahara Analytics API!"}

@app.get("/prompts")
def get_prompts() -> Dict:
    """Returns a list of available prompts."""
    return {"prompts": analyzer.prompts}

@app.post("/analyze")
def analyze_risk(generate_request: GenerateRequest) -> Dict:
    """Endpoint to perform credit risk analysis."""
    prompt = analyzer.get_prompt(generate_request.prompt)
    response = analyzer.analyze(generate_request.message, prompt)
    return {"analysis": response}