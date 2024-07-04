import streamlit as st
import sys
import os
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now you can import your modules
from pages import home, pdf_analysis, website_crawl, file_upload_prompts

from config import APP_TITLE, APP_ICON,PAGES

def main():
    st.set_page_config(layout='wide', page_title=APP_TITLE, page_icon=APP_ICON)
    
    # Appliquer les styles CSS
    from utils import apply_custom_css, hide_streamlit_elements
    apply_custom_css()
    hide_streamlit_elements()

    # Navigation sidebar
    st.sidebar.title("Navigation")
    
    selected_page = st.sidebar.radio("Aller à", list(PAGES.keys()))

    # Afficher la page sélectionnée
    PAGES[selected_page]()

if __name__ == "__main__":
    main()