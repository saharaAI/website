import streamlit as st
from services.llm_service import AITaskOrchestrator



def main():
    st.set_page_config(page_title="Agent App", layout="wide")
    orchestrator = AITaskOrchestrator()
    orchestrator.main()

    