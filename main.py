"""
PubMed Research Assistant

This module provides a system for querying PubMed for medical research articles
and generating responses based on the retrieved information using a language model.

The system consists of the following main components:
1. PubMedFetcher: Retrieves articles from PubMed based on keywords.
2. KeywordGenerator: Generates relevant keywords from a given question.
3. ResponseGenerator: Generates a response to the original question based on the retrieved articles.
4. ResearchPipeline: Orchestrates the entire process from question to answer.


Author: Gaurav Shrivastav
Version: 1.0.0
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from pymed import PubMed
from haystack import Pipeline, Document
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import component
from haystack.utils import Secret

# Load environment variables
load_dotenv()

@component
class PubMedFetcher:
    """
    A component for fetching articles from PubMed based on given queries.
    """

    def __init__(self, tool: str, email: str):
        """
        Initialize the PubMedFetcher.

        Args:
            tool (str): The name of the tool using PubMed API.
            email (str): The email address associated with the PubMed API usage.
        """
        self.pubmed = PubMed(tool=tool, email=email)

    @staticmethod
    def _documentize(article: Any) -> Document:
        """
        Convert a PubMed article to a Haystack Document.

        Args:
            article (Any): A PubMed article object.

        Returns:
            Document: A Haystack Document containing the article information.
        """
        return Document(content=article.abstract, meta={'title': article.title, 'keywords': article.keywords})

    @component.output_types(articles=List[Document])
    def run(self, queries: List[str]) -> Dict[str, List[Document]]:
        """
        Fetch articles from PubMed based on the given queries.

        Args:
            queries (List[str]): A list of search queries.

        Returns:
            Dict[str, List[Document]]: A dictionary containing the fetched articles as Haystack Documents.
        """
        cleaned_queries = queries[0].strip().split('\n')
        articles = []
        try:
            for query in cleaned_queries:
                response = self.pubmed.query(query, max_results=1)
                documents = [self._documentize(article) for article in response]
                articles.extend(documents)
        except Exception as e:
            print(f"Error fetching articles: {e}")
            print(f"Couldn't fetch articles for queries: {queries}")
        return {'articles': articles}

class KeywordGenerator:
    """
    A component for generating keywords from a given question using a language model.
    """

    def __init__(self, model_name: str, api_token: str):
        """
        Initialize the KeywordGenerator.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            api_token (str): The Hugging Face API token.
        """
        self.llm = HuggingFaceTGIGenerator(model_name, token=Secret.from_token(api_token))
        self.llm.warm_up()
        self.prompt_builder = PromptBuilder(template=self.PROMPT_TEMPLATE)

    PROMPT_TEMPLATE = """
    Your task is to convert the following question into 3 keywords that can be used to find relevant medical research papers on PubMed.
    Here is an example:
    question: "What are the latest treatments for major depressive disorder?"
    keywords:
    Antidepressive Agents
    Depressive Disorder, Major
    Treatment-Resistant depression
    ---
    question: {{ question }}
    keywords:
    """

    def generate(self, question: str) -> List[str]:
        """
        Generate keywords from the given question.

        Args:
            question (str): The input question.

        Returns:
            List[str]: A list of generated keywords.
        """
        prompt = self.prompt_builder.run(question=question)
        output = self.llm.run(prompt=prompt['prompt'])
        return output['replies'][0].strip().split('\n')

class ResponseGenerator:
    """
    A component for generating a response to a question based on retrieved articles using a language model.
    """

    def __init__(self, model_name: str, api_token: str):
        """
        Initialize the ResponseGenerator.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            api_token (str): The Hugging Face API token.
        """
        self.llm = HuggingFaceTGIGenerator(model_name, token=Secret.from_token(api_token))
        self.llm.warm_up()
        self.prompt_builder = PromptBuilder(template=self.PROMPT_TEMPLATE)
        
    PROMPT_TEMPLATE = """
    Answer the question truthfully based on the given documents.
    If the documents don't contain an answer, use your existing knowledge base.

    q: {{ question }}
    Articles:
    {% for article in articles %}
      {{article.content}}
      keywords: {{article.meta['keywords']}}
      title: {{article.meta['title']}}
    {% endfor %}
    """

    def generate(self, question: str, articles: List[Document]) -> str:
        """
        Generate a response to the question based on the given articles.

        Args:
            question (str): The input question.
            articles (List[Document]): A list of relevant Haystack Documents.

        Returns:
            str: The generated response.
        """
        prompt = self.prompt_builder.run(question=question, articles=articles)
        output = self.llm.run(prompt=prompt['prompt'], generation_kwargs={"max_new_tokens": 500})
        return output['replies'][0]

class ResearchPipeline:
    """
    A pipeline that orchestrates the entire process of answering medical research questions.
    """

    def __init__(self, pubmed_tool: str, pubmed_email: str, model_name: str, api_token: str):
        """
        Initialize the ResearchPipeline.

        Args:
            pubmed_tool (str): The name of the tool using PubMed API.
            pubmed_email (str): The email address associated with the PubMed API usage.
            model_name (str): The name of the Hugging Face model to use.
            api_token (str): The Hugging Face API token.
        """
        self.fetcher = PubMedFetcher(pubmed_tool, pubmed_email)
        self.keyword_generator = KeywordGenerator(model_name, api_token)
        self.response_generator = ResponseGenerator(model_name, api_token)

    def ask(self, question: str) -> str:
        """
        Process a question and generate a response based on PubMed articles.

        Args:
            question (str): The input question.

        Returns:
            str: The generated response.
        """
        keywords = self.keyword_generator.generate(question)
        articles = self.fetcher.run(keywords)['articles']
        num_of_articles = len(articles)
        response = self.response_generator.generate(question, articles)
        return response, num_of_articles

def main():
    """
    Main function to set up the ResearchPipeline.
    """
    # Load configuration
    huggingface_token = os.getenv('HUGGINGFACE_API_KEY')
    # print(huggingface_token)
    pubmed_tool = os.getenv('PUBMED_TOOL', 'ResearchAssistant')
    pubmed_email = os.getenv('PUBMED_EMAIL', 'research@example.com')
    model_name = os.getenv('MODEL_NAME', 'mistralai/Mixtral-8x7B-Instruct-v0.1')

    # Initialize and return the pipeline
    return ResearchPipeline(pubmed_tool, pubmed_email, model_name, huggingface_token)

if __name__ == "__main__":
    pipeline = main()
    
    # Example usage
    question = "What are the most current treatments for post-acute COVID aka PACS or long COVID?"
    response, num_of_article = pipeline.ask(question)

    print(f"Question: {question}")
    print(f"Response: {response}")
    print(f'Num of articles: {num_of_article}')