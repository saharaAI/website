import streamlit as st
import streamlit.components.v1 as components

st.title("SCRIBD Downloader")

scribd_url = st.text_input("Enter Scribd URL", placeholder="Enter Scribd URL", key="scribd_input")

if st.button("Download PDF"):
    if scribd_url:
        # Construct the URL with the Scribd URL as a query parameter
        download_url = f"https://fast1-1-p9793557.deta.app/download_pdf?url_scribd={scribd_url}"

        # Display a link to trigger the download
        st.markdown(f"[Download PDF]({download_url})")
    else:
        st.warning("Please enter a Scribd URL.")

# Signature
components.html("""
    <div id="signature" style="margin-top: 20px; text-align: center;">
        made with <span style="color: #ff69b4;">♡</span> by Ayoub Abraich | 
        <a href="https://www.linkedin.com/in/ayoub-abraich/" target="_blank">LinkedIn</a> | 
        <a href="mailto:abraich.jobs@gmail.com">Email</a>
    </div>
""")