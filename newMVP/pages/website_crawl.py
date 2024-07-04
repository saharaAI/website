import streamlit as st
from services.website_crawler import crawl_website

def main():
    st.markdown("# Analyse de Site Web 🕸️")
    
    url = st.text_input("Entrez l'URL du site web à analyser")
    depth = st.slider("Profondeur de l'analyse", 1, 5, 3)
    
    if st.button("Lancer l'analyse"):
        if url:
            with st.spinner("Analyse en cours..."):
                result = crawl_website(url, depth)
            
            st.success("Analyse terminée!")
            st.write(result)
        else:
            st.error("Veuillez entrer une URL valide.")

if __name__ == "__main__":
    main()