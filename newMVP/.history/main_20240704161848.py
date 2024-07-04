import streamlit as st


from config import APP_TITLE, APP_ICON,PAGES
from app.utils import apply_custom_css, hide_streamlit_elements

def main():
    st.set_page_config(layout='wide', page_title=APP_TITLE, page_icon=APP_ICON)
    
    # Appliquer les styles CSS
    apply_custom_css()
    hide_streamlit_elements()

    # Navigation sidebar
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox("Select Page", PAGES.keys())
    PAGES[page]()   
    

if __name__ == "__main__":
    main()