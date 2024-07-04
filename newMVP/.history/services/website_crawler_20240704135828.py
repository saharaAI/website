import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def crawl_website(url, max_depth=3):
    visited = set()
    
    def crawl(url, depth):
        if depth > max_depth or url in visited:
            return {}
        
        visited.add(url)
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraire les informations de la page
            title = soup.title.string if soup.title else "Pas de titre"
            links = [urljoin(url, link.get('href')) for link in soup.find_all('a')]
            
            # Récursivement crawler les liens
            subpages = {}
            for link in links:
                if link.startswith(url):  # Rester sur le même domaine
                    subpages.update(crawl(link, depth + 1))
            
            return {url: {"title": title, "subpages": subpages}}
        
        except Exception as e:
            return {url: {"error": str(e)}}
    
    return crawl(url, 0)

# Vous pouvez ajouter d'autres fonctions d'analyse de site web ici