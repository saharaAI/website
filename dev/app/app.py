import streamlit as st
import requests
import json
from datetime import datetime
import geocoder

st.set_page_config(layout='wide', page_title='Sahara Analytics', page_icon='📄')
from pdf_ana import main as pdf_analysis_main
from website_crawl import main as website_crawl_main
from agent_app import main as agent_app_main

# Function to get visitor's IP address
def get_visitor_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        ip_data = json.loads(response.text)
        return ip_data['ip']
    except:
        return 'Unknown'

# Defining the function to get location from IP address
def get_location(ip_address):

    try:
        location = geocoder.ip(ip_address)
        if location.latlng:
            country = location.country
            city = location.city
            lat, lng = location.latlng
            return (country, city, lat, lng)
        else:
            raise ValueError("Could not get location for the given IP address.")
    except Exception as e:
        print(f"Error: {e}")
        return None


# Function to log IP address
def log_ip_address_long_lat(ip_address):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    country, city, lat, lng = get_location(ip_address)
    with open('ip_log.txt', 'a') as f:
        f.write(f"{timestamp}: {ip_address } - Country: {country}, City: {city}, Latitude: {lat}, Longitude: {lng}\n")

# Get and log the IP address at the start of the session
if 'ip_logged' not in st.session_state:
    visitor_ip = get_visitor_ip()
    log_ip_address_long_lat(visitor_ip)
    st.session_state.ip_logged = True

# Hide Streamlit elements
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Define CSS for the app
page_style = """
<style>
.big-font {
    font-size: 30px !important;
    font-weight: bold;
}
.medium-font {
    font-size: 20px !important;
}
.centered {
    text-align: center;
}
.service {

    background-color: #6e166d;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.service h3 {
    color: #fff;
    margin-bottom: 10px;
}

.service p {
    color: #fff;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

def main_page():
    st.markdown('<p class="big-font centered">Sahara Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="medium-font centered">Solutions de Pointe en Gestion du Risque pour les Banques Mauritaniennes</p>', unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("## Nos Services Adaptés au Marché Mauritanien")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="service">
        <h3>Analyse Automatisée du Risque</h3>
        <p>Évaluez la solvabilité des clients avec précision grâce à nos modèles de Machine Learning, adaptés aux spécificités du marché mauritanien.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="service">
        <h3>Détection de la Fraude</h3>
        <p>Protégez votre institution contre les activités frauduleuses en temps réel, avec des algorithmes conçus pour détecter les schémas de fraude locaux.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="service">
        <h3>Gestion de Portefeuille</h3>
        <p>Optimisez vos stratégies de prêt et améliorez la rentabilité avec des outils adaptés au contexte économique mauritanien.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="service">
        <h3>Scoring de Crédit Personnalisé</h3>
        <p>Développez des modèles de scoring adaptés aux réalités socio-économiques de la Mauritanie pour une évaluation plus précise des risques.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## Pourquoi Choisir Sahara Analytics en Mauritanie ?")
    st.markdown("""
    - Expertise locale : Notre équipe comprend des experts du marché financier mauritanien.
    - Données adaptées : Nos modèles intègrent des données spécifiques à l'économie mauritanienne.
    - Support multilingue : Services disponibles en arabe, français et anglais.
    - Conformité : Nos solutions respectent les réglementations de la Banque Centrale de Mauritanie.
    - Flexibilité : Intégration facile avec les systèmes bancaires existants.
    """)

    st.sidebar.markdown("# Accueil 🏠")

def pdf_analysis_page():
    st.markdown("# PDF Analysis 📄")
    st.sidebar.markdown("# PDF Analysis 📄")
    pdf_analysis_main()



# Dictionary mapping page names to their respective functions
page_names_to_funcs = {
    "Accueil": main_page,
    "Analyse PDF": pdf_analysis_page,
    "Website Crawl": website_crawl_main,
    "LLM Agents": agent_app_main,
}

def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Aller à", list(page_names_to_funcs.keys()))
    page_names_to_funcs[selected_page]()

if __name__ == "__main__":
    main()
