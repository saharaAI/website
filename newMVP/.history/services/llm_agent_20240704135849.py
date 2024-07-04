import openai
from config import LLM_MODEL, OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def process_query(query):
    try:
        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Vous êtes un assistant IA spécialisé dans l'analyse financière et la gestion des risques pour les banques mauritaniennes."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Erreur lors du traitement de la requête : {str(e)}"

# Vous pouvez ajouter d'autres fonctions liées à l'agent LLM ici