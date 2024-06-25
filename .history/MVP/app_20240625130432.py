import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import json
import google.generativeai as genai
import re
import random
# --- Configuration ---

# API Key (replace with your actual key)
KEYs = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU" ,"AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]

KEY = random.choice(KEYs)
genai.configure(api_key=KEY)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Sahara Analytics - MVP",
    page_icon=":bank:",
    layout="wide",
)

# Entêtes et introduction
st.title("Sahara Analytics - Solutions Personnalisées de Gestion du Risque de Crédit")
st.markdown("---")

# Barre latérale pour les paramètres
st.sidebar.title("Paramètres")
#api_key = st.sidebar.text_input("Clé API Gemini", type="password")

def get_response(user_input):
        generation_config = {
                    "temperature": 0.8,  # Adjust for creativity (0.2 - more focused, 1.0 - more creative)
                    "top_p": 0.95,      # Controls the diversity of the generated text
                    "top_k": 40,        # Limits the next token choices to the top 'k' probabilities
                    "max_output_tokens": 4096,  # Adjust based on expected report length
                }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(
            history=[{"role": "user", "parts": [user_input]}],
        )
        return chat_session.send_message(user_input)

# Fonction pour interagir avec l'API Gemini
def generate_text(prompt):
    if not api_key:
        st.warning("Veuillez saisir votre clé API Gemini dans la barre latérale.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "max_tokens": 500,  # Ajustez la longueur de la réponse selon vos besoins
    }

    response = requests.post("https://api.gemini.google.com/v1/generate_text", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    else:
        st.error(f"Erreur API Gemini : {response.status_code} - {response.text}")
        return None


# --- Analyse de Documents ---
def analyse_de_documents():
    st.header("Analyse de Documents")
    document_text = st.text_area("Collez le texte du document ici :", height=200)

    if st.button("Analyser le Document"):
        if document_text:
            prompt = f"Extraire les informations clés de ce document financier pour l'évaluation du risque de crédit : {document_text}"
            response = generate_text(prompt)

            if response:
                st.markdown(response)
        else:
            st.warning("Veuillez saisir le texte du document.")


# --- Notation du Risque de Crédit ---
def notation_du_risque():
    st.header("Notation du Risque de Crédit")

    # Exemple de champs de saisie pour les données d'entrée du modèle
    revenu_annuel = st.number_input("Revenu annuel (€)", min_value=0)
    montant_pret = st.number_input("Montant du prêt (€)", min_value=0)
    duree_pret = st.number_input("Durée du prêt (mois)", min_value=1)
    historique_credit = st.selectbox("Historique de crédit", ["Excellent", "Bon", "Moyen", "Mauvais"])

    if st.button("Calculer le Score de Crédit"):
        # Logique pour calculer le score de crédit (remplacez par votre modèle)
        # Exemple simple (à remplacer par un modèle de ML/LLM plus avancé)
        score = (revenu_annuel / montant_pret) * duree_pret

        if historique_credit == "Excellent":
            score *= 1.2
        elif historique_credit == "Bon":
            score *= 1.1
        elif historique_credit == "Mauvais":
            score *= 0.8

        st.success(f"Le score de crédit estimé est : {score:.2f}")


# --- Surveillance du Risque ---
def surveillance_du_risque():
    st.header("Surveillance du Risque")

    # Données d'exemple pour les graphiques
    data = {
        "Mois": ["Janvier", "Février", "Mars", "Avril", "Mai"],
        "Taux de Défaut": [0.02, 0.015, 0.025, 0.018, 0.022],
        "Volume de Prêts": [1000000, 1200000, 1100000, 1300000, 1250000],
    }
    df = pd.DataFrame(data)

    # Création de graphiques
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df["Mois"], df["Taux de Défaut"], "r-", label="Taux de Défaut")
    ax2.bar(df["Mois"], df["Volume de Prêts"], alpha=0.5, label="Volume de Prêts")

    ax1.set_ylabel("Taux de Défaut", color="r")
    ax2.set_ylabel("Volume de Prêts (€)", color="b")
    plt.xlabel("Mois")
    plt.legend()

    st.pyplot(fig)


# --- Tests de Résistance ---
def tests_de_resistance():
    st.header("Tests de Résistance")

    # Exemple de paramètres de simulation
    taux_chomage = st.slider("Taux de chômage (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    taux_interet = st.slider("Taux d'intérêt (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.25)

    if st.button("Exécuter la Simulation"):
        # Logique pour exécuter les tests de résistance (remplacez par votre modèle)
        # Exemple simple (à remplacer par une simulation plus complexe)
        perte_prevue = montant_pret * (taux_chomage / 100) * (taux_interet / 100)
        st.warning(f"Perte prévue sur le prêt : {perte_prevue:.2f}€")

# --- Logique principale de l'application ---
sections = [
    "Analyse de Documents",
    "Notation du Risque de Crédit",
    "Surveillance du Risque",
    "Tests de Résistance",
]

selected_section = st.sidebar.selectbox("Choisissez une section :", sections)

if selected_section == "Analyse de Documents":
    analyse_de_documents()
elif selected_section == "Notation du Risque de Crédit":
    notation_du_risque()
elif selected_section == "Surveillance du Risque":
    surveillance_du_risque()
elif selected_section == "Tests de Résistance":
    tests_de_resistance()