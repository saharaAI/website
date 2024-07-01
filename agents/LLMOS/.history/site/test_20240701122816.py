# app.py (Streamlit app)
import streamlit as st
import requests

st.title("SCRIBD Downloader")

html_form = """
<!DOCTYPE html>
<html lang="en">
<body>
    <form id="pdfForm" action="http://localhost:5000/download" method="post"> 
        <input type="text" id="url_scribd" name="url_scribd" placeholder="Enter Scribd URL" required>
        <button type="submit">Download PDF</button>
    </form>
</body>
</html>
"""

st.markdown(html_form, unsafe_allow_html=True)

if st.button("Process Download"):
    scribd_url = st.session_state.get("url_scribd", None) 
    if scribd_url:
        response = requests.post("http://localhost:5000/download", data={"url_scribd": scribd_url})
        # ... (Handle response from Flask, maybe provide a download link)
    else:
        st.warning("Please enter a Scribd URL in the form and click 'Process Download'.")


# server.py (Flask app)
from flask import Flask, request, send_file

app = Flask(__name__)

@app.route("/download", methods=["POST"])
def download_pdf():
    scribd_url = request.form.get("url_scribd")
    if scribd_url:
        # Your download logic here (using scribd_url)
        # ... (Example: Fetch PDF, save it temporarily)
        return send_file("path_to_downloaded.pdf")
    else:
        return "Scribd URL not provided", 400

if __name__ == "__main__":
    app.run(debug=True) 