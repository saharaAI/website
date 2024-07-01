import streamlit as st
from pdf_ana import main

def main_page():
    st.markdown("# Welcome to the Home Page!")

def page2():
    st.markdown("PDF Analysis")
    main()

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "Main Page": main_page,
    "PDF Analysis": page2,
    "Page 3": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
