import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


from .home import main as home_main
from .pdf_analysis import main as pdf_analysis_main
from .website_crawl import main as website_crawl_main
from .agent_app import main as agent_app_main
from .file_upload_prompts import main as file_upload_prompts_main

__all__ = ['home_main', 'pdf_analysis_main', 'website_crawl_main', 'agent_app_main', 'file_upload_prompts_main']