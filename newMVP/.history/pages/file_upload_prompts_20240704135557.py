import streamlit as st
from services.data_processor import process_uploaded_file

def main():
    st.markdown("# Téléchargement de Fichiers et Prompts 📁")
    
    uploaded_file = st.file_uploader("Choisissez un fichier à analyser", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        prompt = st.text_area("Entrez votre prompt pour l'analyse")
        
        if st.button("Analyser"):
            if prompt:
                with st.spinner("Analyse en cours..."):
                    result = process_uploaded_file(uploaded_file, prompt)
                
                st.success("Analyse terminée!")
                st.write(result)
            else:
                st.error("Veuillez entrer un prompt pour l'analyse.")

if __name__ == "__main__":
    main()