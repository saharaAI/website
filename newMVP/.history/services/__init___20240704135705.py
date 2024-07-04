import openai
from config import LLM_MODEL, OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def process_query(query):
    try:
        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Vous êtes un assistant IA spécialisé dans l'analyse financière et la gestion