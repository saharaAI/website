import streamlit as st

def apply_custom_css():
    custom_css = """
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
    st.markdown(custom_css, unsafe_allow_html=True)

def hide_streamlit_elements():
    hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

# Vous pouvez ajouter d'autres fonctions utilitaires ici