
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import json
import google.generativeai as genai
import re
import random
import plotly.graph_objects as go

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

# --- Classes pour l'application ---

class DocumentAnalyzer:
    def __init__(self):
        pass

    def analyze_document(self, document_text):
        prompt = f"Extraire les informations clés de ce document financier pour l'évaluation du risque de crédit : {document_text}"
        response = get_response(prompt)
        return response.text

class CreditScoringModel:
    def __init__(self):
        pass

    def calculate_credit_score(self, revenu_annuel, montant_pret, duree_pret, historique_credit):
        # Placeholder - Remplacez par un modèle de ML/LLM plus avancé
        score = (revenu_annuel / montant_pret) * duree_pret

        if historique_credit == "Excellent":
            score *= 1.2
        elif historique_credit == "Bon":
            score *= 1.1
        elif historique_credit == "Mauvais":
            score *= 0.8

        return score

class RiskMonitoringDashboard:
    def __init__(self):
        self.data = {
            "Mois": ["Janvier", "Février", "Mars", "Avril", "Mai"],
            "Taux de Défaut": [0.02, 0.015, 0.025, 0.018, 0.022],
            "Volume de Prêts": [1000000, 1200000, 1100000, 1300000, 1250000],
        }
        self.df = pd.DataFrame(self.data)

    def display_interactive_plot(self):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.df["Mois"], 
                y=self.df["Taux de Défaut"], 
                mode='lines+markers', 
                name='Taux de Défaut'
            )
        )
        fig.add_trace(
            go.Bar(
                x=self.df["Mois"], 
                y=self.df["Volume de Prêts"], 
                name='Volume de Prêts', 
                yaxis='y2', 
                opacity=0.5
            )
        )

        fig.update_layout(
            xaxis=dict(title="Mois"),
            yaxis=dict(title="Taux de Défaut", color='blue'),
            yaxis2=dict(title="Volume de Prêts (€)", color='red', overlaying='y', side='right'),
            legend=dict(x=0, y=1.1),
            margin=dict(l=50, r=50, t=80, b=80)
        )

        st.plotly_chart(fig, use_container_width=True)

class StressTestingSimulator:
    def __init__(self):
        pass

    def run_simulation(self, montant_pret, taux_chomage, taux_interet):
        # Placeholder - Remplacez par une simulation plus complexe
        perte_prevue = montant_pret * (taux_chomage / 100) * (taux_interet / 100)
        return perte_prevue

# --- Fonctions utilitaires ---

def get_response(user_input):
    generation_config = {
        "temperature": 0.8,  # Adjust for creativity 
        "top_p": 0.95,      # Controls the diversity of the generated text
        "top_k": 40,        # Limits the next token choices 
        "max_output_tokens": 4096,  # Adjust based on expected length
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [user_input]}],
    )
    return chat_session.send_message(user_input)


# --- Initialisation des instances de classe ---
document_analyzer = DocumentAnalyzer()
credit_scoring_model = CreditScoringModel()
risk_monitoring_dashboard = RiskMonitoringDashboard()
stress_testing_simulator = StressTestingSimulator()

# --- Logique principale de l'application ---
sections = [
    "Analyse de Documents",
    "Notation du Risque de Crédit",
    "Surveillance du Risque",
    "Tests de Résistance",
]

selected_section = st.sidebar.selectbox("Choisissez une section :", sections)

if selected_section == "Analyse de Documents":
    st.header("Analyse de Documents")
    document_text = st.text_area("Collez le texte du document ici :", height=200)

    if st.button("Analyser le Document"):
        if document_text:
            response_text = document_analyzer.analyze_document(document_text)
            st.markdown(response_text)
        else:
            st.warning("Veuillez saisir le texte du document.")

elif selected_section == "Notation du Risque de Crédit":
    st.header("Notation du Risque de Crédit")
    revenu_annuel = st.number_input("Revenu annuel (€)", min_value=0)
    montant_pret = st.number_input("Montant du prêt (€)", min_value=0)
    duree_pret = st.number_input("Durée du prêt (mois)", min_value=1)
    historique_credit = st.selectbox("Historique de crédit", ["Excellent", "Bon", "Moyen", "Mauvais"])

    if st.button("Calculer le Score de Crédit"):
        score = credit_scoring_model.calculate_credit_score(
            revenu_annuel, montant_pret, duree_pret, historique_credit
        )
        st.success(f"Le score de crédit estimé est : {score:.2f}")

elif selected_section == "Surveillance du Risque":
    risk_monitoring_dashboard.display_interactive_plot()

elif selected_section == "Tests de Résistance":
    st.header("Tests de Résistance")
    montant_pret = st.number_input("Montant du prêt (€)", min_value=0)
    taux_chomage = st.slider("Taux de chômage (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    taux_interet = st.slider("Taux d'intérêt (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.25)

    if st.button("Exécuter la Simulation"):
        perte_prevue = stress_testing_simulator.run_simulation(
            montant_pret, taux_chomage, taux_interet
        )
        st.warning(f"Perte prévue sur le prêt : {perte_prevue:.2f}€")