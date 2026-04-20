# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Synthetic interactive conversation workload generator.

Generates multi-turn conversation requests mimicking the paper's
workload pattern: users with 3-8 rounds of conversation, varying
prompt lengths (128-1024 tokens) and response lengths (32-256 tokens).
"""

import argparse
import json
import os
import random
from pathlib import Path

# Seed phrases for generating realistic-looking prompts
PROMPT_TEMPLATES = [
    "Can you explain how {topic} works in simple terms?",
    "I'm writing a paper about {topic}. Can you summarize the key points?",
    "What are the main differences between {topic_a} and {topic_b}?",
    "Please help me understand the concept of {topic}.",
    "I need to learn about {topic} for my project. Where should I start?",
    "Could you provide a detailed analysis of {topic}?",
    "What is the history behind {topic}?",
    "How does {topic} relate to {topic_a}?",
    "Explain the importance of {topic} in modern society.",
    "What are the pros and cons of {topic}?",
    "Can you give me examples of {topic} in practice?",
    "What should I know about {topic} before getting started?",
    "How would you compare {topic} with traditional approaches?",
    "What are the latest developments in {topic}?",
    "Could you break down {topic} step by step?",
    "I'm confused about {topic}. Can you clarify?",
    "What are the common misconceptions about {topic}?",
    "How can I apply {topic} to solve real-world problems?",
    "What resources would you recommend for learning {topic}?",
    "Can you walk me through a typical use case of {topic}?",
]

TOPICS = [
    "machine learning", "neural networks", "natural language processing",
    "computer vision", "reinforcement learning", "transformer models",
    "attention mechanisms", "generative AI", "large language models",
    "fine-tuning", "prompt engineering", "retrieval augmented generation",
    "vector databases", "embedding models", "tokenization",
    "gradient descent", "backpropagation", "convolutional networks",
    "recurrent neural networks", "transfer learning",
    "data preprocessing", "feature engineering", "model evaluation",
    "hyperparameter tuning", "cross-validation", "regularization",
    "dropout techniques", "batch normalization", "residual connections",
    "self-supervised learning", "contrastive learning", "knowledge distillation",
]

FOLLOW_UPS = [
    "That makes sense. Can you elaborate on the technical details?",
    "Interesting! What are the limitations of this approach?",
    "Could you provide a concrete example?",
    "How does this compare to alternative methods?",
    "What are the computational requirements?",
    "Is this approach scalable to larger datasets?",
    "Can you explain the math behind this?",
    "What are the best practices for implementation?",
    "How would this work in a production environment?",
    "Are there any recent papers on this topic?",
    "What are the common pitfalls to avoid?",
    "How do you handle edge cases?",
    "Could you show me a code example?",
    "What performance improvements can I expect?",
    "Is this suitable for real-time applications?",
]


def generate_conversation(
    user_id: str,
    rng: random.Random,
    num_rounds: int | None = None,
) -> list[dict]:
    """Generate a multi-turn conversation for a user."""
    if num_rounds is None:
        num_rounds = rng.randint(3, 8)

    conversation = []
    topic = rng.choice(TOPICS)
    topic_a = rng.choice([t for t in TOPICS if t != topic])

    for round_idx in range(num_rounds):
        if round_idx == 0:
            # First turn: use a prompt template
            prompt = rng.choice(PROMPT_TEMPLATES).format(
                topic=topic, topic_a=topic_a, topic_b=rng.choice(TOPICS)
            )
            prompt_tokens = rng.randint(128, 512)
            output_tokens = rng.randint(64, 256)
        else:
            # Follow-up turns
            prompt = rng.choice(FOLLOW_UPS)
            # Add context from previous rounds
            prompt_tokens = rng.randint(64, 256)
            output_tokens = rng.randint(32, 128)

        conversation.append({
            "user_id": user_id,
            "round": round_idx,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        })

    return conversation


def generate_workload(
    num_users: int = 50,
    output_path: str = "data/interactive_workload.json",
    seed: int = 42,
) -> None:
    """Generate a synthetic interactive conversation workload."""
    rng = random.Random(seed)
    workload = []

    for i in range(num_users):
        user_id = f"user_{i:04d}"
        conversation = generate_conversation(user_id, rng)
        workload.extend(conversation)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(workload, f, indent=2)

    print(f"Generated {len(workload)} requests from {num_users} users")
    print(f"Output: {output_path}")

    # Print statistics
    prompt_lens = [r["prompt_tokens"] for r in workload]
    output_lens = [r["output_tokens"] for r in workload]
    rounds = {}
    for r in workload:
        rounds[r["user_id"]] = max(rounds.get(r["user_id"], 0), r["round"] + 1)

    print(f"\nWorkload Statistics:")
    print(f"  Total requests:     {len(workload)}")
    print(f"  Total users:        {num_users}")
    print(f"  Avg rounds/user:    {sum(rounds.values()) / len(rounds):.1f}")
    print(f"  Prompt tokens:      min={min(prompt_lens)}, max={max(prompt_lens)}, avg={sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"  Output tokens:      min={min(output_lens)}, max={max(output_lens)}, avg={sum(output_lens)/len(output_lens):.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic conversation workload")
    parser.add_argument("--num-users", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/interactive_workload.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_workload(args.num_users, args.output, args.seed)
