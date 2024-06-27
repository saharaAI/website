import google.generativeai as genai
import json
import random
from unstructured.partition.auto import partition
from app.config import Config 
import os
PROMPTS_FILE = os.path.abspath("/home/ayoub/Bureau/projects/startup credit risk scoring/website/flaskapp/app/models/prompts.json")  # Replace with actual path
with open(PROMPTS_FILE, "r") as f:
    self.prompts = json.load(f)
class CreditRiskAnalyzer:
    """A class to encapsulate the credit risk analysis logic."""

    def __init__(self, config: Config = Config):
        """Initializes the analyzer with the given configuration."""
        self.config = config
        genai.configure(api_key=random.choice(self.config.API_KEYS))
        self.model = genai.GenerativeModel(
            model_name=self.config.MODEL_NAME,
            generation_config=self.config.GENERATION_CONFIG
        )
        with open("prompts.json", "r") as f:
            self.prompts = json.load(f)

    def get_prompt(self, prompt_request: dict) -> str:
        """Retrieves the selected prompt with optional context."""
        prompt_key = str(prompt_request['number'])
        if prompt_key not in self.prompts:
            raise ValueError("Invalid prompt number")

        prompt = self.prompts[prompt_key]
        if prompt_request.get('context'):
            prompt += f"\nCONTEXT: {prompt_request['context']}"
        return prompt

    def analyze(self, message: str, prompt: str) -> str:
        """Performs credit risk analysis."""
        chat_session = self.model.start_chat(
            history=[{"role": "user", "parts": [prompt]}]
        )
        response = chat_session.send_message(message)
        return response.text

    def process_uploaded_file(self, file_path: str) -> str:
        """Processes the uploaded file to extract content."""
        elements = partition(filename=file_path)
        file_content = "\n\n".join([str(el) for el in elements] + [""])
        return file_content