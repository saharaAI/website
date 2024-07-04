import pandas as pd
import openai
from app.config import LLM_MODEL, API_KEY

def process_uploaded_file(uploaded_file, prompt):
    try:
        # Lire le fichier
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            return "Format de fichier non supporté"

        # Obtenir un résumé des données
        summary = df.describe().to_string()

        # Utiliser l'API OpenAI pour analyser les données selon le prompt
        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Vous êtes un assistant IA spécialisé dans l'analyse de données financières."},
                {"role": "user", "content": f"Voici un résumé des données : {summary}\n\nAnalysez ces données selon l'instruction suivante : {prompt}"}
            ]
        )

        analysis = response.choices[0].message['content']

        return {
            "résumé_des_données": summary,
            "analyse": analysis
        }

    except Exception as e:
        return f"Erreur lors du traitement du fichier : {str(e)}"

# Vous pouvez ajouter d'autres fonctions de traitement de données ici