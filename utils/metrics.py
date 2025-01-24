# utils/metrics.py

import numpy as np
import evaluate
import json
from utils.faiss_manager import query_faiss
from utils.ollama_interface import generate_answer_with_ollama

def calculate_bleu(generated_answer, reference_answer):
    """Calculate BLEU score for the generated answer."""
    bleu_metric = evaluate.load("bleu")
    result = bleu_metric.compute(predictions=[generated_answer], references=[reference_answer])
    return result["bleu"]

def calculate_rouge(generated_answer, reference_answer):
    """Calculate ROUGE scores for the generated answer."""
    rouge_metric = evaluate.load("rouge")
    result = rouge_metric.compute(predictions=[generated_answer], references=[reference_answer])
    return result

def exact_match(generated_answer, reference_answer):
    """Calculate Exact Match score."""
    return 1 if generated_answer.strip().lower() == reference_answer.strip().lower() else 0

def update_progress(progress_bar, step, total):
    """Updates the progress bar in Streamlit."""
    progress_bar.progress(step / total)

def save_metrics_to_file(metrics, avg_metrics, custom_prompt, top_k, filename="batch_evaluation_results.json"):
    """Save metrics and average metrics to a JSON file, appending to the file if it exists."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    except FileNotFoundError:
        data = []

    # Avoid duplicate prompts
    for entry in data:
        if entry.get("Custom Prompt") == custom_prompt:
            return  # Skip saving if the prompt already exists

    results = {
        "Custom Prompt": custom_prompt,
        "Top K Retrieved": top_k,
        "Metrics": metrics,
        "Average Metrics": avg_metrics
    }

    data.append(results)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def load_saved_prompts_with_metrics(filename="batch_evaluation_results.json"):
    """Function to load saved prompts from the results file"""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        prompts_with_metrics = {
            entry.get("Custom Prompt"): {
                "Top K Retrieved": entry.get("Top K Retrieved"),
                "Average Metrics": entry.get("Average Metrics")
            }
            for entry in data
        }
        return prompts_with_metrics
    except FileNotFoundError:
        return {}

def evaluate_pipeline_with_custom_prompt(custom_prompt, questions, answers, documents, index, model, top_k=5, update_progress=None):
    """Evaluate the RAG pipeline using a custom prompt applied to all queries."""
    metrics = []
    total_queries = len(questions)

    for i, (question, reference_answer) in enumerate(zip(questions, answers)):
        # Retrieve documents
        retrieved_docs = query_faiss(index, question, model, documents, top_k)
        context = "\n".join(retrieved_docs)

        # Replace placeholder in the custom prompt
        prompt = custom_prompt.replace("<retrieved_docs>", context).replace("<query>", question)

        # Generate an answer
        generated_answer = generate_answer_with_ollama(prompt, model_name="llama3.2")

        # Calculate metrics
        bleu = calculate_bleu(generated_answer, reference_answer)
        rouge = calculate_rouge(generated_answer, reference_answer)
        em = exact_match(generated_answer, reference_answer)

        metrics.append({
            "Question": question,
            "Generated Answer": generated_answer,
            "BLEU": bleu,
            "ROUGE-1": rouge['rouge1'],
            "ROUGE-2": rouge['rouge2'],
            "ROUGE-L": rouge['rougeL'],
            "Exact Match": em
        })

        # Update progress bar if callback is provided
        if update_progress:
            update_progress(i + 1, total_queries)

    # Calculate average metrics
    avg_metrics = {
        "Average BLEU": np.mean([m["BLEU"] for m in metrics]),
        "Average ROUGE-1": np.mean([m["ROUGE-1"] for m in metrics]),
        "Average ROUGE-2": np.mean([m["ROUGE-2"] for m in metrics]),
        "Average ROUGE-L": np.mean([m["ROUGE-L"] for m in metrics]),
        "Average Exact Match": np.mean([m["Exact Match"] for m in metrics])
    }


    return metrics, avg_metrics
