import streamlit as st

def main():
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

if __name__ == "__main__":
    main()