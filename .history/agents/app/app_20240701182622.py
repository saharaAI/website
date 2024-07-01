import streamlit as st
from pdf_ana import main as pdf_analysis_main

# --- CONFIG ---
st.set_page_config(layout='wide', page_title='Sahara Analytics', page_icon='📄')

# Hide Streamlit elements
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- STYLE ---
page_style = """
<style>
/* --- Typography --- */
h1, h2, h3, p {
    font-family: 'Arial', sans-serif; 
}

.big-font {
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 1rem;
}

.medium-font {
    font-size: 1.5rem;
    line-height: 1.6;
}

/* --- Layout --- */
.centered {
    text-align: center;
}

.service {
    background-color: #6e166d;
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Add a subtle shadow */
    transition: transform 0.2s; /* Add a hover effect */
}

.service:hover {
    transform: scale(1.05);
}

.service h3 {
    color: white; 
}

/* --- Images --- */
.feature-image {
    max-width: 100%;
    height: auto;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# --- CONTENT ---

def main_page():
    """Displays the content for the main (home) page."""

    st.markdown('<h1 class="big-font centered">Sahara Analytics</h1>', unsafe_allow_html=True)
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
    - **Expertise locale:** Notre équipe comprend des experts du marché financier mauritanien.
    - **Données adaptées:** Nos modèles intègrent des données spécifiques à l'économie mauritanienne.
    - **Support multilingue:** Services disponibles en arabe, français et anglais.
    - **Conformité:** Nos solutions respectent les réglementations de la Banque Centrale de Mauritanie.
    - **Flexibilité:** Intégration facile avec les systèmes bancaires existants.
    """)

def pdf_analysis_page():
    """Displays the content for the PDF analysis page."""
    st.markdown("# Analyse PDF 📄")
    pdf_analysis_main()

def page3():
    """Placeholder for a third page."""
    st.markdown("# Page 3 🎉") 

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Aller à", list(page_names_to_funcs.keys()))

# --- PAGE ROUTING ---
page_names_to_funcs = {
    "Accueil": main_page,
    "Analyse PDF": pdf_analysis_page,
    "Page 3": page3,
}
page_names_to_funcs[page_selection]()