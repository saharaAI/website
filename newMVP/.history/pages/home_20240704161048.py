import streamlit as st

def main():
    # call html file
    with open('main.html', 'r') as f:
        st.markdown(f.read(), unsafe_allow_html=True)




