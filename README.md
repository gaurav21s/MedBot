# PubMed Research Assistant

## Overview

PubMed Research Assistant is a tool that uses the PubMed database and language models to answer medical research questions.

## Features

- Keyword generation from questions
- PubMed article retrieval
- Answer generation using language models
- Streamlit web interface

## Installation

1. Clone the repository:
```shell
git clone https://github.com/yourusername/pubmed-research-assistant.git
cd pubmed-research-assistant
```

2. Create a virtual environment:
```shell
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```
3. Install dependencies:
```shell
pip install -r requirements.txt
```

4. Create a `.env` file:
```python
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
PUBMED_TOOL=ResearchAssistant
PUBMED_EMAIL=your_email@example.com
MODEL_NAME=mistralai/Mixtral-8x7B-Instruct-v0.1
```

## Usage

### Streamlit App
```shell
streamlit run app.py
```
### Python Script

```python
from main import ResearchPipeline

pipeline = ResearchPipeline(pubmed_tool, pubmed_email, model_name, huggingface_token)
question = "What are the most current treatments for post-acute COVID?"
answer = pipeline.ask(question)
print(answer)
```


## Contributing
Contributions welcome. Please submit a Pull Request.

### Acknowledgements

* PubMed
* Hugging Face
* Haystack
* Streamlit