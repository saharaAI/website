import streamlit as st
import pandas as pd
import pytesseract  # For OCR (install with: pip install pytesseract)
from PIL import Image # For OCR (install with: pip install Pillow)
try:
    from PyPDF2 import PdfReader # For PDF handling (install with: pip install PyPDF2) 

except ImportError:
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfpagecontent import PDFPageContent

from utils import get_response  # Assuming you have a utils.py for your Gemini API call
import json


class DocumentProcessor:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # Path to tesseract executable

    def upload_document(self):
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "xlsx", "csv", "jpg", "png"])
        return uploaded_file

    def extract_text(self, uploaded_file):
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1]

            if file_extension == "pdf":
                try: # Try using PyPDF2 first
                    pdf_reader = PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    return text
                except: # Fallback to pdfminer for problematic PDFs
                    parser = PDFParser(uploaded_file)
                    document = PDFDocument(parser)

                    text = ""
                    for page in document.get_pages():
                        content = PDFPageContent.create_content(page)
                        text += content.get_text()
                    return text

            elif file_extension in ["xlsx", "csv"]:
                df = pd.read_excel(uploaded_file) if file_extension == "xlsx" else pd.read_csv(uploaded_file)
                return df.to_string()

            elif file_extension in ["jpg", "png"]:
                image = Image.open(uploaded_file)
                text = pytesseract.image_to_string(image)
                return text 

            else:
                st.warning("File type not supported.")
                return ""
        else:
            return ""

class FinancialDataExtractor:
    def __init__(self):
        pass

    def extract_data(self, text):
        if text: 
            prompt = (
                f"Analyze this financial text and extract data in JSON format:\n\n"
                f"{text}\n\n"
                f"Desired JSON: \n"
                f"{{'company_name': '', 'revenue': , 'net_income': , 'debt': , ... }}"
            )
            response = get_response(prompt)  
            
            # --- Add JSON Parsing Logic Here ---
            try: 
                # Attempt to parse the response.text as JSON
                extracted_data = json.loads(response.text)
            except json.JSONDecodeError as e:
                # Handle cases where Gemini doesn't return valid JSON
                st.error(f"Error parsing JSON from Gemini: {e}")
                st.error(f"Raw response: {response.text}") 
                extracted_data = {} # Return empty dictionary on error

            return extracted_data
        else:
            st.warning("No text to extract data from.")
            return {} 