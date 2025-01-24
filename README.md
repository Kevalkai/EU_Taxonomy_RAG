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

---

# Future Improvements and Enhancements

This project provides a robust foundation for Retrieval-Augmented Generation (RAG). However, there are several opportunities to improve the model’s performance and usability:

---

## 1. Using a More Powerful Model
**Current Limitation**: The project uses Ollama’s `llama3.2`, which is a relatively lightweight language model.  
**Improvement**: Replace `llama3.2` with a more advanced language model, such as **GPT-4** or other large foundation models, to improve contextual understanding and the quality of generated answers.  
**Steps to Implement**:
1. Replace the API calls to Ollama with the new model's API endpoint.
2. Adapt the app to the input/output schema of the new model.
3. Validate the model's performance using the existing evaluation metrics (BLEU, ROUGE, Exact Match).

---

## 2. Incorporating Human Feedback
**Current Context** Evaluating models that generate answers based on multiple retrieved documents using metrics like ROUGE or BLEU is challenging because these metrics rely on the generated text being close to a predefined ground truth. However, when multiple documents contribute to the generation process, the output is often highly diverse and may include valid paraphrases, additional context, or alternative formulations that deviate from the reference answer. Consequently, such outputs might score poorly on these metrics despite being accurate and informative. These metrics, while useful for comparing text similarity, fail to capture semantic correctness or relevance, limiting their utility in evaluating complex, retrieval-augmented generative tasks.

**Purpose**: Improve model performance by leveraging user feedback to guide fine-tuning or enhance retrieval and generation mechanisms.
**Implementation Ideas**:
- Add a feedback mechanism to the app that allows users to rate generated answers as:
  - **Correct**
  - **Partially Correct**
  - **Incorrect**
- Log user feedback alongside metadata (question, retrieved documents, generated answer, and expected answer).
- Use this feedback to improve the model over time.

**Steps to Implement**:
1. **Collect Feedback**:
   - Add a feedback form in the app using Streamlit widgets.
   - Save feedback in a log file or database for future processing.
2. **Analyze Feedback**:
   - Periodically review feedback to identify recurring issues.
   - Refine prompts, retrieval strategies, or other app logic.
3. **Fine-Tune the Model**:
   - Train the language model using curated datasets derived from user feedback.

---

## 3. Optimizing the Retrieval Process
**Current Limitation**: The FAISS index retrieves documents based on embeddings generated by `all-MiniLM-L6-v2`.  
**Improvement**: Experiment with alternative embedding models, which may provide better semantic matching and improve retrieval quality.  
**Steps to Implement**:
1. Evaluate multiple embedding models for retrieval tasks.
2. Compare retrieval quality using metrics like BLEU and ROUGE on a validation set.
3. Replace the current embedding model with the one yielding the best results.

---



By implementing these enhancements, the project can achieve better retrieval accuracy, higher-quality answer generation, and greater alignment with user expectations.


## Thank You
By Kevin Al-Kai
