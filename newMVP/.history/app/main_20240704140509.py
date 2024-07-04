import streamlit as st
from pages import home, pdf_analysis, website_crawl, agent_app, file_upload_prompts
from config import APP_TITLE, APP_ICON

def main():
    st.set_page_config(layout='wide', page_title=APP_TITLE, page_icon=APP_ICON)
    
    # Appliquer les styles CSS
    from utils import apply_custom_css, hide_streamlit_elements
    apply_custom_css()
    hide_streamlit_elements()

    # Navigation sidebar
    st.sidebar.title("Navigation")
    pages = {
        "Accueil": home.main,
        "Analyse PDF": pdf_analysis.main,
        "Website Crawl": website_crawl.main,
        "LLM Agents": agent_app.main,
        "Data Upload > Prompts": file_upload_prompts.main
    }
    selected_page = st.sidebar.radio("Aller à", list(pages.keys()))

    # Afficher la page sélectionnée
    pages[selected_page]()

if __name__ == "__main__":
    main()