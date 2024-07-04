import streamlit as st
from typing import List, Dict
import io
import markdown
from services.pdf_analyzer import process_uploaded_file, save_uploaded_file
from services.llm_service import get_gemini_response

def export_chat(chat_history: List[Dict[str, str]], format: str) -> io.BytesIO:
    md_content = "# Chat History\n\n"
    for message in chat_history:
        role = message["role"]
        content = message.get("parts", [message.get("content", "")])[0]
        md_content += f"## {role.capitalize()}\n\n{content}\n\n"

    if format == "md":
        return io.BytesIO(md_content.encode())
    elif format == "html":
        html_content = markdown.markdown(md_content)
        return io.BytesIO(html_content.encode())

def main():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    
    if 'doc_content' not in st.session_state:
        st.session_state.doc_content = ""

    st.sidebar.title("Configuration")
    prompt = st.sidebar.text_area('Invite de système', value='Agir comme un expert en analyse de documents.')
    uploaded_file = st.sidebar.file_uploader('Télécharger un fichier PDF', type=['pdf'])

    if uploaded_file and not st.session_state.file_processed:
        with st.spinner('Traitement du fichier en cours...'):
            file_path = save_uploaded_file(uploaded_file)
            st.session_state.doc_content = process_uploaded_file(file_path)
            st.session_state.file_processed = True
        st.sidebar.success("Fichier téléchargé et traité avec succès!")

    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            content = message.get("parts", [message.get("content", "")])[0]
            st.write(content)
            if message["role"] == "model":
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    pass
                with col2:
                    md_buffer = export_chat([message], "md")
                    st.download_button(
                        label="📥 MD",
                        data=md_buffer.getvalue(),
                        file_name=f"response_{i}.md",
                        mime="text/markdown",
                        key=f"md_{i}"
                    )
                with col3:
                    html_buffer = export_chat([message], "html")
                    st.download_button(
                        label="📥 HTML",
                        data=html_buffer.getvalue(),
                        file_name=f"response_{i}.html",
                        mime="text/html",
                        key=f"html_{i}"
                    )

    user_input = st.chat_input('Posez votre question sur le document')
    if user_input:
        st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner('Génération de la réponse...'):
                response = get_gemini_response(
                    user_input=user_input,
                    prompt=prompt,
                    context=st.session_state.doc_content,
                    history=st.session_state.chat_history
                )
            st.write(response)
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                pass
            with col2:
                md_buffer = export_chat([{"role": "model", "parts": [response]}], "md")
                st.download_button(
                    label="📥 MD",
                    data=md_buffer.getvalue(),
                    file_name=f"response_{len(st.session_state.chat_history)}.md",
                    mime="text/markdown",
                    key=f"md_{len(st.session_state.chat_history)}"
                )
            with col3:
                html_buffer = export_chat([{"role": "model", "parts": [response]}], "html")
                st.download_button(
                    label="📥 HTML",
                    data=html_buffer.getvalue(),
                    file_name=f"response_{len(st.session_state.chat_history)}.html",
                    mime="text/html",
                    key=f"html_{len(st.session_state.chat_history)}"
                )
        
        st.session_state.chat_history.append({"role": "model", "parts": [response]})

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Exporter toute la conversation en MD'):
            md_buffer = export_chat(st.session_state.chat_history, "md")
            st.download_button(
                label="Télécharger MD",
                data=md_buffer.getvalue(),
                file_name="chat_history.md",
                mime="text/markdown"
            )
    with col2:
        if st.button('Exporter toute la conversation en HTML'):
            html_buffer = export_chat(st.session_state.chat_history, "html")
            st.download_button(
                label="Télécharger HTML",
                data=html_buffer.getvalue(),
                file_name="chat_history.html",
                mime="text/html"
            )

    if st.button('Effacer la conversation'):
        st.session_state.chat_history = []
        st.session_state.file_processed = False
        st.session_state.doc_content = ""
        st.experimental_rerun()

if __name__ == "__main__":
    main()