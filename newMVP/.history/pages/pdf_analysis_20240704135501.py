import streamlit as st
from services.pdf_analyzer import analyze_pdf

def main():
    st.markdown("# Analyse PDF 📄")
    
    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Analyser le PDF"):
            with st.spinner("Analyse en cours..."):
                result = analyze_pdf(uploaded_file)
            
            st.success("Analyse terminée!")
            st.write(result)

if __name__ == "__main__":
    main()