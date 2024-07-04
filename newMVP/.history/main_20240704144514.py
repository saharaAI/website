import streamlit as st

# Now import your modules
from pages import home, pdf_analysis, website_crawl, file_upload_prompts# Now you can import your modules

from app.config import APP_TITLE, APP_ICON,PAGES
from app.utils import apply_custom_css, hide_streamlit_elements

def main():
    st.set_page_config(layout='wide', page_title=APP_TITLE, page_icon=APP_ICON)
    
    # Appliquer les styles CSS
    apply_custom_css()
    hide_streamlit_elements()

    # Navigation sidebar
    st.sidebar.title("Navigation")
    
    selected_page = st.sidebar.radio("Aller à", list(PAGES.keys()))

    # Afficher la page sélectionnée
    PAGES[selected_page]()

if __name__ == "__main__":
    main()