#!/usr/bin/env python
"""
consistency_test.py

This script compares the consistency between Hugging Face (HF) and vLLM model outputs.
It tests for:
  1. Sentence similarity (using a SentenceTransformer encoder)
  2. Perplexity (computed from token logprobs)

vLLM is assumed to be served via an OpenAI-compatible API endpoint. User provides:
  --model_path: path or name for the HF model (e.g., "tokenizer_model_path")
  --served_model_name: model name served via vLLM (e.g., "llama3.1-8b")
  --vllm_url: the URL of the vLLM API endpoint (e.g., http://localhost:8000/v1/completions)

Example:
  python consistency_test.py --model_path tokenizer_model_path --served_model_name llama3.1-8b --vllm_url http://localhost:8000/v1/completions
"""

import argparse
import random
import numpy as np
import torch
import requests
import warnings
import math

from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# -----------
# Setup seed
# -----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed = 0
set_seed(seed)


# ---------------------------
# Sentence similarity function
# ---------------------------
def calculate_sentence_similarity(hf_sentence, vllm_sentence, encoder_model="all-MiniLM-L6-v2"):
    st_model = SentenceTransformer(encoder_model)
    hf_embedding = st_model.encode(hf_sentence, convert_to_tensor=True)
    vllm_embedding = st_model.encode(vllm_sentence, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(hf_embedding, vllm_embedding).item()
    return similarity


# ----------------------------------------
# HF: Generation, logprobs and perplexity
# ----------------------------------------
def run_hf_inference(prompt, model, tokenizer, device, max_tokens):
    """
    Generate a response using HF with greedy decoding, compute token-level logprobs,
    and calculate perplexity from the logprobs.

    - `perplexity` & `logprobs`: Computed on input + output tokens
    - `response_text`: Only output tokens are included

    Returns:
    - response_text: The generated response text (only output part).
    - perplexity: Perplexity score for the entire sequence.
    """
    encoding = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).to(device)

    # Tokenize input prompt
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    input_length = input_ids.shape[1]  # Length of the input prompt

    # Generate output using greedy decoding
    output_ids = model.generate(
        input_ids, attention_mask=attention_mask, max_new_tokens=max_tokens,
        do_sample=False, pad_token_id=tokenizer.eos_token_id
    )

    # Extract only the generated (output) tokens
    generated_ids = output_ids[0][input_length:]  # Exclude input prompt tokens
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Forward pass to obtain logits & loss (for perplexity calculation)
    with torch.no_grad():
        outputs = model(output_ids, labels=output_ids, attention_mask=torch.ones_like(output_ids).to(device))

    # Compute perplexity from loss
    loss = outputs.loss  # CrossEntropyLoss already averaged over tokens
    perplexity = torch.exp(loss).item()

    return response_text, perplexity


# ----------------------------------------
# vLLM: API call for generation, logprobs and perplexity
# ----------------------------------------
def run_vllm_inference(prompt, model_name, vllm_url, max_tokens, timeout=1000):
    """
    Calls the vLLM API to generate a response with logprobs, then computes perplexity.
    
    Returns:
    - response_text: The generated response text.
    - perplexity: Perplexity score computed from logprobs.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "echo": True,
        "logprobs": 1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "add_special_tokens": False
    }
    response = requests.post(vllm_url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    # Extract response text
    response_text = data['choices'][0]['text']
    if response_text.startswith(prompt):
        response_text = response_text[len(prompt):].strip()

    # Extract log probabilities & token IDs
    token_logprobs = data['choices'][0]['logprobs']['token_logprobs']

    perplexity = math.exp(-np.mean([lp for lp in token_logprobs if lp is not None]))

    return response_text, perplexity


# -----------------------------
# Load example prompts from HF
# -----------------------------
def load_example_prompts(num_prompts, min_length, max_length):
    """
    Loads a number of examples from a Hugging Face dataset.
    Here we use the OpenAssistant/oasst1 dataset as an example.
    Only prompts with at least `min_length` words are kept.
    """
    try:
       dataset = load_dataset("OpenAssistant/oasst1", split="train", trust_remote_code=True) # OpenAssistant/oasst1 heegyu/kowiki-sentences
    except Exception as e:
        print("Error loading dataset. Please check the dataset name.")
        raise e
    
    prompts = []
    for example in dataset:
        prompt = example.get("sentence") or example.get("text")

        if prompt:
            prompt_length = len(prompt.split())
            if min_length <= prompt_length <= max_length:
                prompts.append(prompt)
        if len(prompts) >= num_prompts:
            break
    return prompts


# -----------------------------
# Main test function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test consistency between HF and vLLM outputs")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or name for HF model (e.g., tokenizer_model_path)")
    parser.add_argument("--served_model_name", type=str, required=True,
                        help="Model name served via vLLM (e.g., llama3.1-8b)")
    parser.add_argument("--vllm_url", type=str, required=True,
                        help="vLLM API URL (e.g., http://localhost:8000/v1/completions)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max new tokens for generation")
    parser.add_argument("--num_prompts", type=int, default=20,
                        help="Number of prompts to test")
    args = parser.parse_args()

    model_path = args.model_path
    served_model_name = args.served_model_name
    vllm_url = args.vllm_url
    max_tokens = args.max_tokens
    num_prompts = args.num_prompts

    # Load example prompts.
    print("Loading example prompts ...")
    prompts = load_example_prompts(num_prompts=num_prompts, min_length=1000, max_length=4096)
    print(f"Loaded {len(prompts)} prompts.")

    # Load HF model and tokenizer.
    print(f"Loading HF model and tokenizer for {model_path} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not hf_tokenizer.pad_token:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto").to(device)
    hf_model.eval()

    # Metrics accumulators.
    total_sentence_similarity = 0.0
    total_hf_perplexity = 0.0
    total_vllm_perplexity = 0.0
    num_tests = len(prompts)

    for idx, prompt in enumerate(prompts):
        print(f"\n--- Prompt {idx+1}/{num_tests} ---")
        print("Prompt (first 100 chars):", prompt[:100], "...")
        
        # HF inference.
        try:
            hf_text, hf_ppl = run_hf_inference(
                prompt, hf_model, hf_tokenizer, device, max_tokens)
        except Exception as e:
            print(f"Error during HF inference for prompt {idx+1}: {e}")
            continue
        
        # vLLM inference.
        try:
            vllm_text, vllm_ppl = run_vllm_inference(
                prompt, served_model_name, vllm_url, max_tokens)
        except Exception as e:
            print(f"Error during vLLM inference for prompt {idx+1}: {e}")
            continue

        print(f"HF response (first 100 chars):", hf_text[:100], "...")
        print(f"vLLM response (first 100 chars):", vllm_text[:100], "...")

        # 1. Sentence similarity.
        sent_sim = calculate_sentence_similarity(hf_text, vllm_text)
        total_sentence_similarity += sent_sim
        print(f"Sentence similarity: {sent_sim:.4f}")

        # 3. Perplexity.
        total_hf_perplexity += hf_ppl
        total_vllm_perplexity += vllm_ppl
        print(f"HF Perplexity: {hf_ppl:.4f}")
        print(f"vLLM Perplexity: {vllm_ppl:.4f}")

    # ---------------
    # Aggregated results
    # ---------------
    print("\n=== Aggregated Results ===")
    avg_sent_sim = total_sentence_similarity / num_tests if num_tests > 0 else 0
    print(f"Average sentence similarity: {avg_sent_sim:.4f}")
    avg_hf_ppl = total_hf_perplexity / num_tests if num_tests > 0 else float('inf')
    avg_vllm_ppl = total_vllm_perplexity / num_tests if num_tests > 0 else float('inf')
    print(f"Average HF Perplexity: {avg_hf_ppl:.4f}")
    print(f"Average vLLM Perplexity: {avg_vllm_ppl:.4f}")


if __name__ == "__main__":
    main()
