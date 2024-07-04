import PyPDF2
from io import BytesIO

def analyze_pdf(uploaded_file):
    try:
        # Lire le fichier PDF
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
        
        # Extraire le texte de toutes les pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Analyser le texte (ceci est un exemple simple, vous pouvez ajouter une analyse plus complexe)
        word_count = len(text.split())
        char_count = len(text)
        
        # Retourner les résultats de l'analyse
        return {
            "nombre_de_pages": len(pdf_reader.pages),
            "nombre_de_mots": word_count,
            "nombre_de_caracteres": char_count,
            "extrait": text[:500] + "..." if len(text) > 500 else text
        }
    except Exception as e:
        return {"error": str(e)}

# Vous pouvez ajouter d'autres fonctions d'analyse PDF ici