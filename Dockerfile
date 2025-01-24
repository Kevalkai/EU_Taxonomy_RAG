# Base image with Python
FROM python:3.11.4-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for Ollama and Python
RUN apt-get update && apt-get install -y \
    wget curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama (Linux version)
RUN wget https://ollama.com/ -O /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pull the required Ollama model
RUN ollama pull llama3.2

# Expose necessary ports
EXPOSE 8501 11434

# Default command to run both servers
CMD ["sh", "-c", "ollama serve & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
