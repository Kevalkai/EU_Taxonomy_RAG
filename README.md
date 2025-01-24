# EU Taxonomy Retrieval-Augmented Generation (RAG) Model



This project implements a **Retrieval-Augmented Generation (RAG)** model to answer questions related to the **EU Taxonomy** using a combination of **FAISS** for document retrieval and **Ollama's Llama 3.2** model for answer generation.

The app provides an intuitive interface to:
- Test single queries.
- Evaluate custom prompts with batch queries.
- Analyze BLEU, ROUGE, and Exact Match metrics for the model's responses.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Ollama Setup](#ollama-setup)
4. [Running the Application](#running-the-application)
5. [Project Structure](#project-structure)
6. [Sample Data](#sample-data)
7. [Custom Prompt Evaluation](#custom-prompt-evaluation)
8. [Known Issues](#known-issues)

---

## Features
1. **Document Retrieval**: Uses **FAISS** to retrieve the most relevant documents for a query.
2. **Answer Generation**: Leverages **Ollama's Llama 3.2** to generate answers based on retrieved documents.
3. **Metrics Evaluation**: Supports evaluation using BLEU, ROUGE, and Exact Match metrics.
4. **Custom Prompts**: Allows users to design and test their own prompts for the RAG pipeline.
5. **Batch Evaluation**: Enables testing over a batch of queries with progress tracking.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/Kevalkai/EU_Taxonomy_RAG.git
cd EU_Taxonomy_RAG
```

### Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Ollama Setup

### Install Ollama
1. Visit [Ollama](https://ollama.com) and follow the instructions to download and install the CLI.
2. Verify the installation:
   ```bash
   ollama --version
   ```

### Pull the Model
Download the **Llama 3.2** model using the following command:
```bash
ollama pull llama3.2
```

### Start the Ollama Server
Run the server locally:
```bash
ollama serve
```

Ensure the server is running on port `11434` (default).

---

## Running the Application

### Start the Streamlit App
After ensuring that the Ollama server is running, start the app with:
```bash
streamlit run app.py
```

By default, the app will run at [http://localhost:8501](http://localhost:8501).

---

## Project Structure

```plaintext
EU_Taxonomy_RAG/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   └── faq_data.json           # Sample FAQ data for testing
├── utils/
│   ├── corpus_loader.py        # Functions to load and process the JSON corpus
│   ├── faiss_manager.py        # Functions to build, query, and manage the FAISS index
│   ├── ollama_interface.py     # Functions to interact with the Ollama API
│   ├── metrics.py              # Evaluation metrics (BLEU, ROUGE, Exact Match)
├── batch_evaluation_results.json  # Saved results of batch evaluations
└── .gitignore                  # Git ignore file
```

---

## Sample Data

The project includes a sample JSON file (`faq_data.json`) with all the questions and answers scrapped from EU Taxonomy FAQ and used to create the FAISS Vector DB. 
**Format**:
```json
[
    {
        "question": "Will the technical screening criteria set out in the Climate Delegated Act be made stricter and updated over time?",
        "answer": "Article 19(5) of the Taxonomy Regulation requires the Commission to regularly review..."
    },
    ...
]
```

---

## Custom Prompt Inference

The app allows users to test custom prompts or use previous generated prompts for the RAG pipeline. Custom prompts can be created with placeholders:
- `<retrieved_docs>`: Represents the retrieved documents for the query.
- `<query>`: Represents the user's question.

**Example Custom Prompt**:
```text
Summarize the following context and provide a concise answer to the question:
Context:
<retrieved_docs>
Question:
<query>
Answer:
```

### Batch Evaluation
1. Navigate to the **Batch Evaluation** page.
2. Enter a custom prompt.
3. Select the number of top documents to retrieve and the number of random samples to evaluate.
4. Run the evaluation to compute metrics.

Metrics are saved to `batch_evaluation_results.json`.

---

## Known Issues

1. **Ollama Server Dependency**:
   - The app requires the Ollama server to be running locally.
   - Ensure the correct model (`llama3.2`) is downloaded.

2. **Performance**:
   - Batch evaluation with large data can take time, especially for high values of `top_k`.

3. **Port Conflicts**:
   - The Ollama server must use the default port (`11434`) unless the app is modified.


---

## Thank You
By Kevin Al-Kai
