# app.py (Main Streamlit Application)

import streamlit as st
import random
import json
from utils.corpus_loader import load_corpus_from_json
from utils.faiss_manager import build_faiss_index, load_faiss_index, query_faiss
from utils.ollama_interface import generate_answer_with_ollama
from utils.metrics import evaluate_pipeline_with_custom_prompt, update_progress, save_metrics_to_file, load_saved_prompts_with_metrics
from sentence_transformers import SentenceTransformer
import subprocess



def is_installed(command):
    """Check if a command is available on the system."""
    try:
        subprocess.run([command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def install_ollama():
    """Install Ollama if not already installed."""
    if not is_installed("ollama"):
        print("Ollama is not installed. Installing now...")
        try:
            subprocess.run(
                ["curl", "-fsSL", "https://ollama.com/install.sh | sh"],
                check=True,
                shell=True
            )
            print("Ollama installation script executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during installation: {e.stderr}")
    else:
        print("Ollama is already installed.")

def pull_llama(version):
    """Pull the specified version of Llama using the full path to Ollama."""
    ollama_path = "./ollama"  # Update this path based on your system
    try:
        print(f"Pulling Llama version {version}...")
        result = subprocess.run(
            [ollama_path, "pull", f"llama{version}"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Llama version {version} pulled successfully:\n{result.stdout}")
    except FileNotFoundError:
        print(f"Error: The command '{ollama_path}' was not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling Llama version {version}:\n{e.stderr}")

def serve_ollama():
    """Start the Ollama server."""
    try:
        print("Starting Ollama server...")
        subprocess.run(
            ["ollama", "serve"],
            check=True,
            text=True
        )
        print("Ollama server is now running.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting Ollama server: {e.stderr}")


# Main Streamlit Interface
def main():
    st.title("RAG approach for the FAQs of the EU Taxonomy Navigator ")


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
            st.markdown("### Single Querie Inference")
            # Single Query Page
            query = st.text_input("Enter your question:")
            top_k = st.slider("Number of documents to retrieve:", 1, 10, 5)

            # Load saved prompts with metrics
            saved_prompts_with_metrics = load_saved_prompts_with_metrics()
            saved_prompts = list(saved_prompts_with_metrics.keys())

            selected_prompt = st.selectbox("Select a saved custom prompt:", ["None"] + saved_prompts)

            if selected_prompt != "None":
                custom_prompt = st.text_area(
                    "Selected Custom Prompt (editable):",
                    selected_prompt
                )

                # Display metrics for the selected prompt
                metrics_info = saved_prompts_with_metrics[selected_prompt]
                st.markdown("### Metrics for Selected Prompt")
                st.write(f"**Top K Retrieved:** {metrics_info['Top K Retrieved']}")
                if metrics_info["Average Metrics"]:
                    avg_metrics = metrics_info["Average Metrics"]
                    st.write(f"**Average BLEU:** {avg_metrics['Average BLEU']:.4f}")
                    st.write(f"**Average ROUGE-1:** {avg_metrics['Average ROUGE-1']:.4f}")
                    st.write(f"**Average ROUGE-2:** {avg_metrics['Average ROUGE-2']:.4f}")
                    st.write(f"**Average ROUGE-L:** {avg_metrics['Average ROUGE-L']:.4f}")
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
            st.markdown("**Evaluate custom prompt applied to a set of queries in the database.**")
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
    install_ollama()

    serve_ollama()
    pull_llama("3.2")
    main()
