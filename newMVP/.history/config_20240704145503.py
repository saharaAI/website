from pages import home, pdf_analysis, website_crawl, file_upload_prompts
import random

# Configuration de l'application
APP_TITLE = 'Sahara Analytics'
APP_ICON = '📄'

# Mapping des pages
PAGES = {
    "Accueil": home.main,
    "Analyse PDF": pdf_analysis.main,
    "Website Crawl": website_crawl.main,
    "LLM Agents": agent_app.main,
    "Data Upload > Prompts": file_upload_prompts.main
}

# Autres configurations
PDF_UPLOAD_PATH = './uploads/pdf/'
WEBSITE_CACHE_PATH = './cache/websites/'


KEYS = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
MODEL_NAME = "gemini-1.5-flash"
UPLOAD_DIR = "uploads"
API_KEY = random.choice(KEYS)
