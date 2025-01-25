# Inspect Evals Examples

# Project Setup Guide

## Prerequisites
- Python 3.x
- Git
- GROQ API key

## Installation

### 1. Clone repository
```bash
git clone https://github.com/AI-Plans/inspect-evals-examples.git
cd inspect-evals-examples
```
### 2. Virtual environment
# Create
```bash
python3 -m venv venv
```

# Activate
```bash
source venv/bin/activate  # Unix/macOS
.\venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment setup

# Create .env file:
```text
GROQ_API_KEY=<your_api_key>
MODEL_NAME='groq/llama-3.1-8b-instant'
PROVIDER='groq'
DATASET=<dataset_name>
```


# Notes

Ensure GROQ API key has credits - or that you're using a free model
Use virtual environment for isolated dependencies
Check requirements.txt for versions
