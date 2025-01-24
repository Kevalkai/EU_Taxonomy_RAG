# utils/ollama_interface.py

import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/chat"  # Replace with your Ollama server's URL if needed

def generate_answer_with_ollama(prompt, model_name="llama3.2"):
    """Generates an answer using the Llama 3.2 model from Ollama."""
    # Make a request to the Ollama server
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": model_name, "messages": [{"role": "user", "content": prompt}]},
        stream=True  # Enable streaming
    )

    if response.status_code != 200:
        raise Exception(f"Error from Ollama API: {response.status_code}, {response.text}")

    # Process the streaming response
    answer = ""
    for line in response.iter_lines(decode_unicode=True):
        if line.strip():  # Skip empty lines
            try:
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    answer += chunk["message"]["content"]  # Append the chunk to the answer
            except json.JSONDecodeError:
                print(f"Warning: Could not decode line: {line}")

    return answer.strip()
