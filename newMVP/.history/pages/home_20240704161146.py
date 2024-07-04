import streamlit as st

def main():
    # call html file
    with open('static/html/home.html', 'r') as f:
        st.markdown(f.read(), unsafe_allow_html=True)




if __name__ == "__main__":
    main()