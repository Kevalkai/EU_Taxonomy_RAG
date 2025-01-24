# utils/corpus_loader.py

import json
import streamlit as st

@st.cache_resource
def load_corpus_from_json(file_path):
    """Loads a text corpus from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Extract questions and answers
    documents = [f"Q: {item['question']} A: {item['answer']}" for item in data]
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    return documents, questions, answers
