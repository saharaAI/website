import streamlit as st
st.set_page_config(layout='wide', page_title='Sahara Analytics', page_icon='📄')

from pdf_ana import main as pdf_analysis_main


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

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

# Dictionary mapping page names to their respective functions
page_names_to_funcs = {
    "Accueil": main_page,
    "Analyse PDF": pdf_analysis_page,
    "Page 3": page3,
}

def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Aller à", list(page_names_to_funcs.keys()))
    page_names_to_funcs[selected_page]()

if __name__ == "__main__":
    main()
