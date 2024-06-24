"""
Streamlit app for PubMed Research Assistant

This script creates a simple web interface using Streamlit to interact with
the PubMed Research Assistant. Users can input a medical research question,
and the app will display the generated response based on PubMed articles.

Author: Gaurav Shrivastav
Version: 1.0.0
"""

import streamlit as st
from dotenv import load_dotenv
import os
import time

# Set page config at the very beginning
st.set_page_config(page_title="AI Medical Research Assistant", page_icon="🏥", layout="wide")

# Load environment variables
load_dotenv()

# Import ResearchPipeline after setting page config
from main import ResearchPipeline

# Custom CSS to enhance the UI
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
local_css("style.css")

def initialize_pipeline():
    """
    Initialize the ResearchPipeline with configuration from environment variables.
    """
    huggingface_token = os.getenv('HUGGINGFACE_API_KEY')
    pubmed_tool = os.getenv('PUBMED_TOOL', 'ResearchAssistant')
    pubmed_email = os.getenv('PUBMED_EMAIL', 'research@example.com')
    model_name = os.getenv('MODEL_NAME', 'mistralai/Mixtral-8x7B-Instruct-v0.1')

    return ResearchPipeline(pubmed_tool, pubmed_email, model_name, huggingface_token)

def main():
    """
    Main function to run the Streamlit app.
    """
    # Sidebar
    st.sidebar.image("HealthcareChatbotArchitecture.png", use_column_width=True)
    st.sidebar.title("AI Medical Research Assistant")
    st.sidebar.info(
        "This app uses advanced AI to help you find answers to medical research questions. "
        "Simply enter your question, and the AI will search PubMed for relevant articles and generate a response."
    )

    # Main content
    st.title("🔬 PubMed Research Assistant")

    # Initialize the pipeline
    pipeline = initialize_pipeline()

    # Create a text input for the user's question
    question = st.text_area("Enter your medical research question:", height=100)

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        search_button = st.button("🔍 Search and Analyze", use_container_width=True)

    if search_button and question:
        with st.spinner("🧠 Researching... This may take a moment."):
            # Use the pipeline to get the answer
            start_time = time.time()
            answer,num_of_articles = pipeline.ask(question)
            end_time = time.time()

        # Display the answer
        st.subheader("📊 Research Summary")
        st.write(answer)

        # Display additional information
        st.subheader("📈 Process Analytics")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
        with col2:
            st.success(f"✅ PubMed articles analyzed: {num_of_articles}")  

    # Add a footer
    st.markdown("---")
    st.markdown(
        "@ Gaurav Shrivastav | "
        "Powered by PubMed, Hugging Face, and Streamlit | "
        "[GitHub](https://github.com/yourusername/medical-research-assistant)"
    )

if __name__ == "__main__":
    main()