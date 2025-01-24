# app.py (Main Streamlit Application)

import streamlit as st
import random
import json
from utils.corpus_loader import load_corpus_from_json
from utils.faiss_manager import build_faiss_index, load_faiss_index, query_faiss
from utils.ollama_interface import generate_answer_with_ollama
from utils.metrics import evaluate_pipeline_with_custom_prompt, update_progress, save_metrics_to_file
from sentence_transformers import SentenceTransformer

# Function to load saved prompts from the results file
def load_saved_prompts(filename="batch_evaluation_results.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        prompts = [entry.get("Custom Prompt") for entry in data] if isinstance(data, list) else [data.get("Custom Prompt")]
        return prompts
    except FileNotFoundError:
        return []

def save_new_prompt(custom_prompt, filename="batch_evaluation_results.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
    except FileNotFoundError:
        data = []

    data.append({"Custom Prompt": custom_prompt})

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# Main Streamlit Interface
def main():
    st.title("Custom Prompt Evaluation")
    st.markdown("**Evaluate single queries or a custom prompt applied to all queries in the database.**")

    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Single Query", "Batch Evaluation"])

    # Load JSON corpus file
    json_path = st.text_input("Enter the path to your JSON corpus file:", "faq_data.json")
    if json_path and "documents" not in st.session_state:
        with st.spinner("Loading corpus and building FAISS index..."):
            documents, questions, answers = load_corpus_from_json(json_path)
            st.session_state["documents"] = documents
            st.session_state["questions"] = questions
            st.session_state["answers"] = answers

            # Load or build FAISS index
            index, embeddings = load_faiss_index()
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Ensure the model is always loaded

            if index is None:
                st.info("No existing FAISS index found. Building a new one...")
                index, embeddings, model = build_faiss_index(documents)
            else:
                st.success("Loaded existing FAISS index.")

            st.session_state["index"] = index
            st.session_state["model"] = model

    # Check if the index is loaded
    if "index" in st.session_state:
        if page == "Single Query":
            # Single Query Page
            query = st.text_input("Enter your question:")
            top_k = st.slider("Number of documents to retrieve:", 1, 10, 5)

            # Load saved prompts
            saved_prompts = load_saved_prompts()
            selected_prompt = st.selectbox("Select a saved custom prompt:", ["None"] + saved_prompts)

            if selected_prompt != "None":
                custom_prompt = st.text_area(
                    "Selected Custom Prompt (editable):",
                    selected_prompt
                )
            else:
                custom_prompt = st.text_area(
                    "Enter your custom prompt (use <retrieved_docs> and <query> as placeholders):",
                    "Answer the following question using the provided context:\nContext:\n<retrieved_docs>\nQuestion: <query>\nAnswer:"
                )

            if query:
                with st.spinner("Retrieving relevant documents..."):
                    retrieved_docs = query_faiss(
                        st.session_state["index"],
                        query,
                        st.session_state["model"],
                        st.session_state["documents"],
                        top_k
                    )

                st.subheader("Retrieved Documents")
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Document {i+1}:**")
                    st.markdown(doc)

                # Generate answer
                with st.spinner("Generating answer..."):
                    context = "\n".join(retrieved_docs)
                    prompt = custom_prompt.replace("<retrieved_docs>", context).replace("<query>", query)
                    answer = generate_answer_with_ollama(prompt)

                st.subheader("Generated Answer")
                st.markdown(answer)

        elif page == "Batch Evaluation":
            st.markdown("### Batch Evaluation")

            # Input for custom prompt
            custom_prompt = st.text_area(
                "Enter your custom prompt (use <retrieved_docs> and <query> as placeholders):",
                "Answer the following question using the provided context:\nContext:\n<retrieved_docs>\nQuestion:\n<query>\nAnswer:"
            )
            top_k = st.slider("Number of documents to retrieve:", 1, 10, 5)
            sample_size = st.number_input("Number of random samples to evaluate:", min_value=1, max_value=len(st.session_state["questions"]), value=5, step=1)
            save_results = st.checkbox("Save evaluation results to a file", value=False)

            if st.button("Run Batch Evaluation"):
                with st.spinner("Running batch evaluation..."):
                    sampled_indices = random.sample(range(len(st.session_state["questions"])), sample_size)
                    sampled_questions = [st.session_state["questions"][i] for i in sampled_indices]
                    sampled_answers = [st.session_state["answers"][i] for i in sampled_indices]

                    progress_bar = st.progress(0)

                    metrics, avg_metrics = evaluate_pipeline_with_custom_prompt(
                        custom_prompt, sampled_questions, sampled_answers,
                        st.session_state["documents"], st.session_state["index"], st.session_state["model"], top_k,
                        update_progress=lambda step, total: update_progress(progress_bar, step, total)
                    )

                    if save_results:
                        save_new_prompt(custom_prompt)
                        save_metrics_to_file(metrics, avg_metrics, custom_prompt, top_k)

                st.subheader("Batch Evaluation Metrics")
                for result in metrics:
                    st.markdown(f"**Question:** {result['Question']}")
                    st.write(f"Generated Answer: {result['Generated Answer']}")
                    st.write(f"BLEU: {result['BLEU']:.4f}")
                    st.write(f"ROUGE-1: {result['ROUGE-1']:.4f}")
                    st.write(f"ROUGE-2: {result['ROUGE-2']:.4f}")
                    st.write(f"ROUGE-L: {result['ROUGE-L']:.4f}")
                    st.write(f"Exact Match: {result['Exact Match']}\n")

                st.subheader("Average Metrics")
                st.write(f"Average BLEU: {avg_metrics['Average BLEU']:.4f}")
                st.write(f"Average ROUGE-1: {avg_metrics['Average ROUGE-1']:.4f}")
                st.write(f"Average ROUGE-2: {avg_metrics['Average ROUGE-2']:.4f}")
                st.write(f"Average ROUGE-L: {avg_metrics['Average ROUGE-L']:.4f}")

                if save_results:
                    st.success("Evaluation results and prompt saved to 'batch_evaluation_results.json'")

if __name__ == "__main__":
    main()
