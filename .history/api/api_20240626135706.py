from typing import Dict, Optional 
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends,status
from pydantic import BaseModel
import google.generativeai as genai
import random
import json
import os
import shutil
from unstructured.partition.auto import partition
# JSONResponse
from fastapi.responses import JSONResponse, FileResponse
# --- Configuration ---

class Config:
    """Configuration class to store API keys and other settings."""
    API_KEYS = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
    MODEL_NAME = "gemini-1.5-flash"
    GENERATION_CONFIG = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    ALLOWED_FILE_EXTENSIONS = {"txt", "pdf", "csv", "xlsx", "md", "docx"}
    UPLOAD_FOLDER = "uploads"

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
        with open("prompts.json", "r") as f:
            self.prompts = json.load(f)

    def get_prompt(self, prompt_request: PromptRequest) -> str:
        """Retrieves the selected prompt with optional context."""
        prompt_key = str(prompt_request.number)  # Convert to string for key access
        if prompt_key not in self.prompts:
            raise HTTPException(status_code=400, detail="Invalid prompt number")

        prompt = self.prompts[prompt_key]
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

    def process_uploaded_file(self, file: UploadFile) -> str:
        """Processes the uploaded file to extract content."""
        file_extension = file.filename.split(".")[-1]
        if file_extension not in self.config.ALLOWED_FILE_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Save the uploaded file locally
        file_path = os.path.join(self.config.UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Remove the uploaded file
        file.file.close()

        elements = partition(filename=file_path)
        file_content = "\n\n".join([str(el) for el in elements] + [""])
        print(file_content)
        return file_content

# --- FastAPI app ---

app = FastAPI(
    title="Sahara Analytics API",
    description="An API for credit risk analysis using Google Gemini.",
    version="0.1.0"
)

analyzer = CreditRiskAnalyzer(Config)  # Initialize the analyzer

@app.exception_handler(Exception)
async def validation_exception_handler(request, exc):
    """Global exception handler for the API."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": f"Error: {str(exc)}"},
    )

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

@app.post("/upload_file/")
async def upload_and_analyze(file: UploadFile = File(...), prompt_number: int = 0):
    """Endpoint to upload a file and analyze its content."""
    try:
        # Process the uploaded file
        file_content = analyzer.process_uploaded_file(file)

        # Example: Use the file content as context for the analysis
        generate_request = GenerateRequest(
            message="Analyze this information.", 
            prompt=PromptRequest(number=prompt_number, context=file_content)
        )
        analysis_result = analyze_risk(generate_request)

        return {"analysis": analysis_result}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
