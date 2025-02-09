# Consistency Test between Hugging Face and vLLM Models

## Overview
`consistency_test.py` is a Python script designed to evaluate the consistency of text generation between a Hugging Face (HF) model and a vLLM model served via an OpenAI-compatible API endpoint. It performs two key comparisons:

1. **Sentence Similarity**: Measures how semantically similar the outputs of the HF and vLLM models are using a SentenceTransformer encoder.
2. **Perplexity**: Evaluates the fluency of generated text by computing token log probabilities.

## Features
- Compares responses from HF and vLLM models for a set of predefined prompts.
- Computes sentence similarity using `SentenceTransformer`.
- Calculates perplexity scores for both HF and vLLM responses.
- Uses an OpenAI-compatible vLLM API for inference.
- Loads example prompts from the Hugging Face dataset (`OpenAssistant/oasst1`).

## Prerequisites
### Required Libraries
Ensure you have the following Python packages installed:
```bash
pip install torch transformers sentence-transformers datasets numpy requests
```

### Hardware & Environment
- A CUDA-compatible GPU is recommended for running HF model inference efficiently.
- The vLLM model should be served using an OpenAI-compatible API.

## Usage
### Command-Line Arguments
```bash
python consistency_test.py --model_path <hf_model_path> --served_model_name <vllm_model_name> --vllm_url <vllm_api_url>
```

### Example
```bash
python consistency_test.py --model_path tokenizer_model_path \
  --served_model_name llama3.1-8b \
  --vllm_url http://localhost:8000/v1/completions
```

### Arguments
- `--model_path`: Path or name for the HF model (e.g., `tokenizer_model_path`).
- `--served_model_name`: Name of the model served via vLLM (e.g., `llama3.1-8b`).
- `--vllm_url`: The URL of the vLLM API endpoint (e.g., `http://localhost:8000/v1/completions`).
- `--max_tokens`: (Optional) Maximum number of new tokens to generate (default: `1024`).
- `--num_prompts`: (Optional) Number of example prompts to test (default: `20`).

## Functionality Breakdown
### 1. Set Random Seed
Ensures reproducibility of results by setting a fixed random seed for NumPy, PyTorch, and Python’s random module.

### 2. Sentence Similarity Calculation
Uses `SentenceTransformer` to encode text from both HF and vLLM models and computes cosine similarity.

### 3. Hugging Face Model Inference
- Generates text using a HF model.
- Computes token log probabilities.
- Calculates perplexity from log probabilities.

### 4. vLLM Model Inference
- Calls the vLLM API using an OpenAI-compatible request format.
- Extracts generated text and token log probabilities.
- Computes perplexity from log probabilities.

### 5. Load Example Prompts
- Uses the `OpenAssistant/oasst1` dataset from Hugging Face.
- Filters prompts based on word length constraints.

### 6. Aggregated Results
At the end of execution, the script reports:
- **Average Sentence Similarity**: Between HF and vLLM outputs.
- **Average HF Perplexity**: Fluency metric for HF model.
- **Average vLLM Perplexity**: Fluency metric for vLLM model.

## Example Output
```
--- Prompt 1/20 ---
Prompt (first 100 chars): The concept of machine learning dates back to ...
HF response (first 100 chars): Machine learning is a subset of artificial ...
vLLM response (first 100 chars): Machine learning has revolutionized ...
Sentence similarity: 0.8945
HF Perplexity: 15.2314
vLLM Perplexity: 16.0456
...
=== Aggregated Results ===
Average sentence similarity: 0.8723
Average HF Perplexity: 14.8921
Average vLLM Perplexity: 15.7634
```

## Notes
- Ensure the vLLM API is running before executing the script.
- If an error occurs during dataset loading, verify the dataset exists and is accessible.
- The script assumes that vLLM outputs OpenAI-compatible JSON responses.