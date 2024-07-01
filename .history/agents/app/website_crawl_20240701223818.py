import streamlit as st
import trafilatura
from trafilatura.spider import focused_crawler
import google.generativeai as genai
from typing import List, Dict
import os
import hashlib
import random
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Constants
KEYS = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
MODEL_NAME = "gemini-1.5-flash"
UPLOAD_DIR = "uploads"
MAX_URLS = 5
MAX_WORKERS = 4

# Configure genai
genai.configure(api_key=random.choice(KEYS))  # Use a random key for better load distribution

class WebScraper:
    def __init__(self, url):
        self.url = url
        self.extracted_text = ""
        self.metadata = {}
        self.crawled_links = []
        self.all_text = ""

    def scrape(self):
        try:
            downloaded = trafilatura.fetch_url(self.url)
            self.extracted_text = trafilatura.extract(downloaded)
            self.all_text += self.extracted_text
        except Exception as e:
            st.error(f"An error occurred during scraping: {e}")

    def crawl(self, max_urls=MAX_URLS):
        try:
            with st.spinner('Crawling in progress...'):
                progress_bar = st.progress(0)
                self.crawled_links, _ = focused_crawler(self.url, max_seen_urls=max_urls)
                
                def process_link(link):
                    try:
                        response = requests.get(link, timeout=10)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        return trafilatura.extract(soup.prettify())
                    except Exception as e:
                        st.warning(f"Error processing {link}: {e}")
                        return ""

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_link = {executor.submit(process_link, link): link for link in self.crawled_links}
                    for i, future in enumerate(as_completed(future_to_link)):
                        self.all_text += future.result()
                        progress_bar.progress((i + 1) / len(self.crawled_links))

            st.success('Crawling completed!')
        except Exception as e:
            st.error(f"An error occurred during crawling: {e}")



class GeminiApp:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }

    def generate_response(self, context: str, num_urls: int) -> str:
        model = genai.GenerativeModel(model_name=self.model_name, generation_config=self.generation_config)
        
        prompt = f"""
        You are an expert data analyst and report generator. You have been given the content from {num_urls} web pages. Your task is to analyze this content and provide a comprehensive report in two parts: an English summary and a structured JSON output.

        Context (excerpt from scraped web pages):
        {context[:7000]}

        Please generate a report with the following structure:

        1. English Summary:
        - Provide an overview of the main topics covered in the scraped content.
        - Identify and explain key themes, trends, or patterns you observe.
        - Highlight any significant data points, statistics, or quotes.
        - Offer insights or conclusions based on the analyzed content.

        2. JSON Output:
        Generate a JSON object with the following structure:
        {{
          "main_topics": ["topic1", "topic2", "topic3"],
          "key_themes": [
            {{
              "theme": "Theme 1",
              "description": "Brief description of theme 1",
              "related_topics": ["topic1", "topic2"]
            }},
            // ... more themes
          ],
          "significant_data_points": [
            {{
              "data_point": "Specific data point or statistic",
              "context": "Brief context or explanation",
              "source": "Source of the data point (if available)"
            }},
            // ... more data points
          ],
          "insights": [
            {{
              "insight": "Key insight or conclusion",
              "explanation": "Brief explanation or supporting evidence"
            }},
            // ... more insights
          ]
        }}

        Ensure that your JSON is valid and properly formatted.

        Begin your response with the English summary, followed by the JSON output.
        """
        
        return model.generate_content(prompt).text


def save_extracted_text(text, filename):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path

def main():
    st.title("Enhanced Web Scraper & Gemini App")

    url = st.text_input("Enter the URL of the website:", value="https://irshad.mr/appels-doffres/")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Scrape, Crawl, and Analyze"):
        if url:
            scraper = WebScraper(url)
            scraper.scrape()
            scraper.crawl(max_urls=MAX_URLS)
            
            if scraper.crawled_links:
                st.subheader("Crawled Links:")
                # as json
                st.json(scraper.crawled_links)

            file_name = f"{hashlib.md5(url.encode()).hexdigest()}.txt"
            file_path = save_extracted_text(scraper.all_text, file_name)
            st.success(f"Extracted text saved to: {file_path}")

            gemini = GeminiApp()
            response = gemini.generate_response(scraper.all_text, len(scraper.crawled_links) + 1)
            
            st.subheader("Gemini Report:")
            
            # Split the response into English summary and JSON parts
            parts = response.split('```json', 1)
            
            if len(parts) == 2:
                english_summary, json_part = parts
                st.markdown("### English Summary")
                st.write(english_summary.strip())
                
                st.markdown("### JSON Output")
                try:
                    json_data = json.loads(json_part.strip().rstrip('`'))
                    st.json(json_data)
                except json.JSONDecodeError:
                    st.error("Failed to parse JSON. Raw output:")
                    st.code(json_part.strip(), language='json')
            else:
                st.write(response)

            st.session_state.chat_history.append({"role": "user", "content": "Generate comprehensive report"})
            st.session_state.chat_history.append({"role": "model", "content": response})

        else:
            st.warning("Please enter a valid URL.")

if __name__ == "__main__":
    main()