import argparse
import re
import numpy as np
import pandas as pd

def load_and_prepare_data(dataset_name, split):
    """Loads the dataset from Hugging Face and converts it to a pandas DataFrame."""
    import datasets
    # Disable progress bar to prevent Windows-specific environment variable limit issues
    datasets.utils.logging.disable_progress_bar()
    
    print(f"Loading dataset '{dataset_name}' (split: '{split}')...")
    ds = datasets.load_dataset(dataset_name, split=split)
    df = ds.to_pandas()
    return df

def extract_contrastive_pairs(df, reward_prefix):
    """Dynamically finds reward/response columns, matches indices, and extracts pos/neg pairs."""
    # Find matching reward columns (e.g. response_{idx}_{reward_prefix})
    reward_pattern = re.compile(rf"^response_(\d+)_(.*{re.escape(reward_prefix)}.*)$")
    
    pairs_mapping = {}  # idx -> reward_col
    for col in df.columns:
        match = reward_pattern.match(col)
        if match:
            idx = int(match.group(1))
            pairs_mapping[idx] = col
            
    if not pairs_mapping:
        raise ValueError(
            f"No reward columns found matching prefix '{reward_prefix}'.\n"
            f"Available columns: {list(df.columns)}"
        )
        
    print(f"Found matching reward columns: {list(pairs_mapping.values())}")
    
    pairs = []
    for _, row in df.iterrows():
        prompt = row['prompt']
        
        valid_pairs = []
        for idx, reward_col in pairs_mapping.items():
            resp_col = f"response_{idx}"
            if resp_col in df.columns:
                resp_val = row[resp_col]
                reward_val = row[reward_col]
                
                # Check for None or NaN values
                if (resp_val is not None and not (isinstance(resp_val, float) and np.isnan(resp_val))) and \
                   (reward_val is not None and not (isinstance(reward_val, float) and np.isnan(reward_val))):
                    valid_pairs.append((resp_val, float(reward_val)))
                    
        if len(valid_pairs) < 2:
            continue
            
        # Find highest score (Positive) and lowest score (Negative)
        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        pos_response, pos_score = valid_pairs[0]
        neg_response, neg_score = valid_pairs[-1]
        
        score_delta = pos_score - neg_score
        
        pairs.append({
            'prompt': prompt,
            'pos_response': pos_response,
            'neg_response': neg_response,
            'pos_score': pos_score,
            'neg_score': neg_score,
            'score_delta': score_delta
        })
        
    return pd.DataFrame(pairs)

def filter_pairs(df, delta_percentile, max_len_diff_ratio):
    """Filters the pairs based on score delta percentile and response length discrepancy."""
    if df.empty:
        return df
        
    # Delta Filter
    threshold = df['score_delta'].quantile(delta_percentile)
    print(f"Score delta threshold at {delta_percentile:.2f} percentile: {threshold:.4f}")
    df = df[df['score_delta'] >= threshold].copy()
    
    if df.empty:
        return df
        
    # Length Filter: |len_pos - len_neg| / max(len_pos, len_neg) <= max_len_diff_ratio
    def get_len_diff_ratio(row):
        len_pos = len(str(row['pos_response']))
        len_neg = len(str(row['neg_response']))
        max_len = max(len_pos, len_neg)
        if max_len == 0:
            return 0.0
        return abs(len_pos - len_neg) / max_len
        
    df['len_diff_ratio'] = df.apply(get_len_diff_ratio, axis=1)
    df = df[df['len_diff_ratio'] <= max_len_diff_ratio].copy()
    
    df = df.drop(columns=['len_diff_ratio'])
    return df

def apply_chat_templates(df, tokenizer_id):
    """Initializes the tokenizer and formats prompt/response pairs with chat templates."""
    from transformers import AutoTokenizer
    
    print(f"Loading tokenizer '{tokenizer_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    # Set standard ChatML fallback template if no template is defined natively
    if tokenizer.chat_template is None:
        print("Warning: Tokenizer does not have a default chat template. Using fallback ChatML template.")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
        )
        
    def format_text(prompt, response):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        # Native template formatting; handles default system prompts internally if dictated by tokenizer config
        return tokenizer.apply_chat_template(messages, tokenize=False)
        
    print("Applying chat template formatting...")
    df['positive_input'] = df.apply(lambda r: format_text(r['prompt'], r['pos_response']), axis=1)
    df['negative_input'] = df.apply(lambda r: format_text(r['prompt'], r['neg_response']), axis=1)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="PersonalLLM Contrastive Dataset Builder")
    parser.add_argument("--dataset-name", type=str, default="namkoong-lab/PersonalLLM", help="Source dataset name")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--reward-model", type=str, required=True, help="Prefix/substring of reward model columns")
    parser.add_argument("--delta-percentile", type=float, default=0.70, help="Score difference percentile threshold (0-1)")
    parser.add_argument("--len-diff-ratio", type=float, default=0.20, help="Maximum allowed length discrepancy ratio")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Tokenizer model ID")
    parser.add_argument("--output", type=str, default="contrastive_dataset.parquet", help="Output filename (.parquet)")
    
    args = parser.parse_args()
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data(args.dataset_name, args.dataset_split)
    print(f"Original dataset size: {len(df)}")
    
    # Step 2: Extract contrastive pairs
    df_pairs = extract_contrastive_pairs(df, args.reward_model)
    print(f"Extracted pairs size: {len(df_pairs)}")
    
    # Step 3: Filter pairs
    df_filtered = filter_pairs(df_pairs, args.delta_percentile, args.len_diff_ratio)
    print(f"Pairs after filtering: {len(df_filtered)}")
    
    # Step 4: Apply chat templates
    df_templated = apply_chat_templates(df_filtered, args.tokenizer)
    
    # Step 5: Save output
    df_templated.to_parquet(args.output, index=False)
    print(f"Successfully saved contrastive preference dataset to '{args.output}'")

if __name__ == "__main__":
    main()
