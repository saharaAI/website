import streamlit as st
from config import APP_TITLE, APP_ICON, PAGES
from app.utils import apply_custom_css, hide_streamlit_elements
import streamlit.components.v1 as components

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    
    # Read the HTML file
    with open('static/html/home.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Apply custom CSS to remove padding
    st.markdown("""
        <style>
        .main {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
            padding-right: 0rem;
        }
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
            padding-right: 0rem;
        }
        </style>
        """, unsafe_allow_html=True)

    # Use streamlit components to render the HTML
    components.html(html_content, height=1000, scrolling=True)

    # Apply custom CSS and hide Streamlit elements
    apply_custom_css()
    hide_streamlit_elements()

    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", list(PAGES.keys()))
    PAGES[page]()

if __name__ == "__main__":
    main()