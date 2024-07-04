import streamlit as st
import streamlit.components.v1 as components

def main():
    
    # Read the HTML file
    with open('static/html/home.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Use streamlit components to render the HTML
    components.html(html_content, height=600, scrolling=True)

if __name__ == "__main__":
    main()
