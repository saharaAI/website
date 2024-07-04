from pages import home, pdf_analysis, website_crawl, agent_app, file_upload_prompts

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

# Configurations spécifiques aux services
PDF_ANALYSIS_MODEL = 'path/to/your/model'
WEBSITE_CRAWL_DEPTH = 3
LLM_MODEL = 'gpt-3.5-turbo'  # ou tout autre modèle que vous utilisez