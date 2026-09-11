import argparse
import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

def load_model_and_tokenizer(model_id, quantization):
    """Loads tokenizer and model with appropriate quantization for memory management."""
    print(f"Loading tokenizer '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Standard ChatML fallback template if no template is defined natively
    if tokenizer.chat_template is None:
        print("Warning: Tokenizer does not have a default chat template. Using fallback ChatML template.")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
        
    print(f"Loading model '{model_id}' (quantization: '{quantization}')...")
    if quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto"
        )
    elif quantization == "8bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map="auto"
        )
    else:  # "none"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
    return model, tokenizer

def get_response_token_indices(tokenizer, input_text, prompt_text):
    """Isolates the sequence indices corresponding to the response part of the templated text."""
    input_ids = tokenizer.encode(input_text)
    prompt_ids = tokenizer.encode(prompt_text)
    
    # Align prompt prefix with input tokens to identify the start of the response
    match_len = 0
    for i in range(min(len(prompt_ids), len(input_ids))):
        if prompt_ids[i] == input_ids[i]:
            match_len += 1
        else:
            break
            
    start_idx = match_len if match_len > 0 else len(prompt_ids)
    return list(range(start_idx, len(input_ids)))

def get_hidden_states(model, tokenizer, text, response_indices, target_layer):
    """Runs a forward pass and extracts pooled hidden states for response tokens at target layers."""
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
    hidden_states = outputs.hidden_states
    
    if target_layer == -1:
        # Extract all layers except the initial embedding layer 0
        layers_to_extract = list(range(1, len(hidden_states)))
    else:
        layers_to_extract = [target_layer]
        
    layer_vectors = {}
    for layer in layers_to_extract:
        # hidden_states[layer] shape: [batch_size=1, seq_len, hidden_dim]
        h = hidden_states[layer][0]  # shape: [seq_len, hidden_dim]
        
        # Filter valid indices within bounds
        valid_indices = [idx for idx in response_indices if idx < h.size(0)]
        if not valid_indices:
            valid_indices = [-1]  # Fallback to last token if indices are empty
            
        h_sliced = h[valid_indices]  # shape: [num_response_tokens, hidden_dim]
        pooled = h_sliced.mean(dim=0)  # shape: [hidden_dim]
        layer_vectors[layer] = pooled
        
    return layer_vectors

def main():
    parser = argparse.ArgumentParser(description="Extract steering vectors from contrastive pairs")
    parser.add_argument("--dataset", type=str, default="contrastive_dataset.parquet", help="Path to parquet dataset")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="HF model ID")
    parser.add_argument("--quantization", type=str, choices=["none", "8bit", "4bit"], default="4bit", help="Quantization level")
    parser.add_argument("--target-layer", type=str, default="-1", help="Target layer to extract (-1 for all layers)")
    parser.add_argument("--output-dir", type=str, default="steering_vectors", help="Directory to save vectors")
    
    args = parser.parse_args()
    
    target_layer_int = int(args.target_layer)
    
    # Load dataset
    df = pd.read_parquet(args.dataset)
    print(f"Loaded dataset with {len(df)} rows.")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.quantization)
    
    # Determine target layers to track
    if target_layer_int == -1:
        if hasattr(model.config, "num_hidden_layers"):
            layers_to_track = list(range(1, model.config.num_hidden_layers + 1))
        else:
            layers_to_track = []
    else:
        layers_to_track = [target_layer_int]
        
    layer_accumulators = {}
    
    # Iterate through prompt/response pairs
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting hidden states"):
        prompt = row['prompt']
        pos_input = row['positive_input']
        neg_input = row['negative_input']
        
        # Format user prompt up to assistant prefix
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        # Isolate token indices for the response parts
        pos_indices = get_response_token_indices(tokenizer, pos_input, prompt_text)
        neg_indices = get_response_token_indices(tokenizer, neg_input, prompt_text)
        
        # Forward pass & hidden states extraction
        pos_vectors = get_hidden_states(model, tokenizer, pos_input, pos_indices, target_layer_int)
        neg_vectors = get_hidden_states(model, tokenizer, neg_input, neg_indices, target_layer_int)
        
        # Dynamically set layer indices if config was missing
        if not layers_to_track:
            layers_to_track = list(pos_vectors.keys())
            
        if not layer_accumulators:
            layer_accumulators = {layer: [] for layer in layers_to_track}
            
        # Accumulate contrastive difference
        for layer in layers_to_track:
            if layer in pos_vectors and layer in neg_vectors:
                delta = pos_vectors[layer] - neg_vectors[layer]
                layer_accumulators[layer].append(delta.detach().cpu())
                
    # Average vectors across the dataset
    print("Averaging steering vectors across dataset...")
    mean_vectors = {}
    for layer, vectors in layer_accumulators.items():
        if vectors:
            mean_vectors[f"layer_{layer}"] = torch.stack(vectors).mean(dim=0).to(torch.float32)
            
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    if target_layer_int == -1:
        output_path = os.path.join(args.output_dir, "steering_vectors_all_layers.pt")
    else:
        output_path = os.path.join(args.output_dir, f"steering_vectors_layer_{target_layer_int}.pt")
        
    torch.save(mean_vectors, output_path)
    print(f"Successfully saved steering vectors to '{output_path}'")

if __name__ == "__main__":
    main()
