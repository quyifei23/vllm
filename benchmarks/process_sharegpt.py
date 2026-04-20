# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Process ShareGPT dataset into a format suitable for Bidaw benchmarking.

Extracts multi-turn conversations, assigns user IDs, and tokenizes
prompts/responses to get accurate token counts.
"""

import argparse
import json
import os
from transformers import AutoTokenizer


def process_sharegpt(
    input_path: str,
    output_path: str,
    model_path: str,
    max_rounds: int = 4,
    max_samples: int = 500,
    seed: int = 42,
) -> None:
    """
    Process ShareGPT JSON into benchmark requests.

    Output format: list of dicts with:
        - user_id: str (derived from conversation id)
        - round: int (0-based turn index)
        - prompt: str (human turn text)
        - prompt_tokens: int
        - output_tokens: int
        - output: str (gpt turn text, for optional validation)
    """
    import random
    rng = random.Random(seed)

    print(f"Loading ShareGPT from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} conversations")

    # Sample conversations to work with
    if len(data) > max_samples:
        data = rng.sample(data, max_samples)
        print(f"  Sampled {max_samples} conversations")

    # Load tokenizer for token counting
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    requests = []
    skipped = 0

    for conv in data:
        conv_id = conv.get("id", "unknown")
        # ShareGPT format: conversations is a list of turns with "from" and "value"
        turns = conv.get("conversations", [])
        if not turns:
            skipped += 1
            continue

        # Extract human/gpt pairs
        pairs = []
        for turn in turns:
            role = turn.get("from", "").lower()
            text = turn.get("value", "").strip()
            if not text:
                continue
            if role in ("human", "user"):
                if pairs and pairs[-1][0] is None:
                    # Previous was human without gpt response, replace
                    pairs[-1] = (text, None)
                else:
                    pairs.append((text, None))
            elif role in ("gpt", "assistant", "bot"):
                if pairs and pairs[-1][1] is None:
                    pairs[-1] = (pairs[-1][0], text)
                else:
                    pairs.append((None, text))

        # Filter valid pairs (must have both prompt and response)
        valid_pairs = [(p, r) for p, r in pairs if p is not None and r is not None]

        if not valid_pairs:
            skipped += 1
            continue

        # Limit rounds
        valid_pairs = valid_pairs[:max_rounds]

        for round_idx, (prompt_text, output_text) in enumerate(valid_pairs):
            prompt_tokens = len(tokenizer.encode(prompt_text))
            output_tokens = len(tokenizer.encode(output_text))

            # Skip degenerate cases
            if prompt_tokens < 4 or output_tokens < 4:
                continue
            if prompt_tokens > 4096 or output_tokens > 2048:
                continue

            requests.append({
                "user_id": f"user_{conv_id.replace('/', '_')}",
                "round": round_idx,
                "prompt": prompt_text,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "output": output_text,
            })

    # Print statistics
    prompt_lens = [r["prompt_tokens"] for r in requests]
    output_lens = [r["output_tokens"] for r in requests]
    user_ids = set(r["user_id"] for r in requests)
    rounds_per_user = {}
    for r in requests:
        rounds_per_user[r["user_id"]] = max(rounds_per_user.get(r["user_id"], 0), r["round"] + 1)

    print(f"\nProcessed workload:")
    print(f"  Total requests:     {len(requests)}")
    print(f"  Unique users:       {len(user_ids)}")
    print(f"  Avg rounds/user:    {sum(rounds_per_user.values()) / max(len(rounds_per_user), 1):.1f}")
    print(f"  Prompt tokens:      min={min(prompt_lens)}, max={max(prompt_lens)}, avg={sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"  Output tokens:      min={min(output_lens)}, max={max(output_lens)}, avg={sum(output_lens)/len(output_lens):.0f}")
    print(f"  Skipped:            {skipped}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(requests, f, indent=2)
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ShareGPT into benchmark workload")
    parser.add_argument("--input", type=str, default="/root/autodl-fs/data/ShareGPT_V3_unfiltered_cleaned_split.json")
    parser.add_argument("--output", type=str, default="data/sharegpt_workload.json")
    parser.add_argument("--model", type=str, default="/root/autodl-fs/model/OPT-6.7B")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    process_sharegpt(args.input, args.output, args.model, args.max_rounds, args.max_samples, args.seed)
