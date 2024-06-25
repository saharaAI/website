
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import random

# --- Configuration ---

# API Keys (replace with your actual keys)
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
import random

def generer_document_aleatoire():
    """Génère un texte simulant un document financier pour tester l'application."""

    entreprises = ["ABC Inc.", "Tech Solutions Ltd.", "Global Enterprises", "Data Analytics Corp."]
    secteurs = ["technologie", "finance", "énergie", "santé"]
    indicateurs = {
        "chiffre d'affaires": {"plage": (1000000, 100000000), "suffixe": "€"},
        "bénéfice net": {"plage": (50000, 1000000), "suffixe": "€"},
        "ratio d'endettement": {"plage": (0.1, 2.0), "suffixe": ""},
        "flux de trésorerie disponible": {"plage": (-200000, 500000), "suffixe": "€"},
    }

    entreprise = random.choice(entreprises)
    secteur = random.choice(secteurs)

    document = f"**Analyse Financière de {entreprise}**\n\n"
    document += f"{entreprise} est une entreprise du secteur {secteur}."

    for indicateur, valeurs in indicateurs.items():
        valeur = random.uniform(valeurs["plage"][0], valeurs["plage"][1])
        document += f"\n- {indicateur.capitalize()} : {valeur:.2f}{valeurs['suffixe']}"

    # Ajouter des phrases aléatoires pour simuler un texte plus réaliste
    phrases_aleatoires = [
        "L'entreprise prévoit une croissance de son chiffre d'affaires de 10% au cours de l'année prochaine.",
        "La direction a mis en place un plan de réduction des coûts pour améliorer la rentabilité.",
        "Des risques liés à la concurrence et à l'évolution du marché sont à prendre en compte.",
        "L'entreprise dispose d'une situation financière solide avec un faible niveau d'endettement.",
    ]
    document += "\n\n" + random.choice(phrases_aleatoires)

    # call gemini api
    prompt = f"Imagine a full writien demand of loan document with key financial indicators and realistic numbers, payements etc in French based on info: {document}"
    
    doc =  get_response(prompt).text
    return doc
class DocumentAnalyzer:
    def __init__(self):
        pass

    def analyze_document(self, document_text):
        """Analyse un document financier et extrait les informations clés."""
        prompt = (
            f"Analyse the following financial document and provide insights relevant to credit risk assessment, "
            f"including key financial indicators, potential risks, and any information that might affect creditworthiness: ALWAYS IN FRENCH \n\n"
            f"{document_text}. in final give info as dictionary"
        )
        response = get_response(prompt)
        return response.text

class CreditScoringModel:
    def __init__(self):
        pass

    def calculate_credit_score(self, revenu_annuel, montant_pret, duree_pret, historique_credit, 
                              autres_dettes=0, nombre_cartes_credit=0, taux_utilisation_credit=0):
        """Calcule un score de crédit en utilisant une logique plus complexe."""
        # Score de base basé sur le ratio dette/revenu
        ratio_dette_revenu = (montant_pret + autres_dettes) / revenu_annuel
        score = (1 - ratio_dette_revenu) * 500 

        # Ajustements en fonction de l'historique de crédit
        if historique_credit == "Excellent":
            score += 150
        elif historique_credit == "Bon":
            score += 100
        elif historique_credit == "Moyen":
            score += 50
        else:
            score -= 50  

        # Ajustements en fonction de l'utilisation du crédit
        score -= taux_utilisation_credit * 2 
        score -= nombre_cartes_credit * 5 

        # Ajustements en fonction de la durée du prêt
        score += duree_pret 

        # Assurer que le score est dans une plage valide
        score = max(0, min(score, 850)) 

        return score

class RiskMonitoringDashboard:
    def __init__(self):
        self.data = {
            "Mois": ["Janvier", "Février", "Mars", "Avril", "Mai"],
            "Taux de Défaut": [0.02, 0.015, 0.025, 0.018, 0.022],
            "Volume de Prêts": [1000000, 1200000, 1100000, 1300000, 1250000],
            "Clients à Risque": [50, 45, 60, 55, 48],
            "Pertes sur Prêts": [10000, 8000, 12000, 9500, 11000]
        }
        self.df = pd.DataFrame(self.data)

    def display_interactive_plot(self):
        """Affiche un tableau de bord interactif avec plusieurs graphiques."""
        col1, col2 = st.columns(2)

        with col1:
            # Graphique 1: Taux de Défaut et Volume de Prêts
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
                title="Taux de Défaut et Volume de Prêts",
                xaxis=dict(title="Mois"),
                yaxis=dict(title="Taux de Défaut", color='blue'),
                yaxis2=dict(title="Volume de Prêts (€)", color='red', overlaying='y', side='right'),
                legend=dict(x=0, y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Graphique 2: Clients à Risque
            fig = go.Figure(data=[go.Pie(labels=self.df["Mois"], values=self.df["Clients à Risque"], hole=.3)])
            fig.update_layout(title="Répartition des Clients à Risque")
            st.plotly_chart(fig, use_container_width=True)

        # Graphique 3: Pertes sur Prêts
        fig = go.Figure(
            data=[go.Bar(x=self.df["Mois"], y=self.df["Pertes sur Prêts"])],
            layout_title_text="Pertes sur Prêts Mensuelles"
        )
        st.plotly_chart(fig, use_container_width=True)

class StressTestingSimulator:
    def __init__(self):
        pass

    def run_simulation(self, montant_pret, taux_chomage, taux_interet, horizon=12, intervalle_confiance=0.95):
        """Exécute une simulation de test de résistance plus avancée."""
        # Simulation simple - à remplacer par un modèle économétrique
        pertes_simulees = []
        for _ in range(1000): 
            perte = montant_pret * (1 + (taux_interet / 100)) 
            for _ in range(horizon):
                choc_chomage = random.uniform(-taux_chomage, taux_chomage) 
                perte *= (1 + choc_chomage / 100) 
            pertes_simulees.append(perte)

        # Calculer les statistiques de perte
        perte_moyenne = sum(pertes_simulees) / len(pertes_simulees)
        perte_max = max(pertes_simulees)

        # Calculer la perte au niveau de confiance spécifié
        perte_intervalle_confiance = sorted(pertes_simulees)[int(intervalle_confiance * len(pertes_simulees))]

        return {
            "Perte Moyenne": perte_moyenne,
            "Perte Maximale": perte_max,
            f"Perte ({int(intervalle_confiance * 100)}% de Confiance)": perte_intervalle_confiance
        }


# --- Fonctions utilitaires ---

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
    document_text = st.text_area("Collez le texte du document ici :", height=200, value=generer_document_aleatoire())

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
    autres_dettes = st.number_input("Autres dettes (€)", min_value=0)
    nombre_cartes_credit = st.number_input("Nombre de cartes de crédit", min_value=0)
    taux_utilisation_credit = st.slider("Taux d'utilisation du crédit (%)", min_value=0, max_value=100, value=0)


    if st.button("Calculer le Score de Crédit"):
        score = credit_scoring_model.calculate_credit_score(
            revenu_annuel, montant_pret, duree_pret, historique_credit,
            autres_dettes, nombre_cartes_credit, taux_utilisation_credit
        )
        st.success(f"Le score de crédit estimé est : **{score:.2f}**")

elif selected_section == "Surveillance du Risque":
    risk_monitoring_dashboard.display_interactive_plot()

elif selected_section == "Tests de Résistance":
    st.header("Tests de Résistance")
    montant_pret = st.number_input("Montant du prêt (€)", min_value=0)
    taux_chomage = st.slider("Variation maximale du taux de chômage (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    taux_interet = st.slider("Taux d'intérêt annuel du prêt (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.25)
    horizon = st.slider("Horizon de la simulation (mois)", min_value=6, max_value=60, value=12, step=6)
    intervalle_confiance = st.slider("Niveau de confiance pour la perte (%)", min_value=50, max_value=99, value=95, step=1)

    if st.button("Exécuter la Simulation"):
        resultats_simulation = stress_testing_simulator.run_simulation(
            montant_pret, taux_chomage, taux_interet, horizon, intervalle_confiance / 100
        )
        st.write("**Résultats de la Simulation:**")
        for key, value in resultats_simulation.items():
            st.write(f"- {key}: **{value:.2f}€**")