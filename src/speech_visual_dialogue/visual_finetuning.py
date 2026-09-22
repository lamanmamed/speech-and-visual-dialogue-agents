"""Training-data formatting and LoRA settings for the visual Oracle."""

from __future__ import annotations

from .visual_dialogue import build_candidate_description


def build_oracle_training_example(
    image,
    question: str,
    answer: str,
    candidates: list[dict],
    target_index: int,
) -> dict:
    description = build_candidate_description(candidates, image.size)
    prompt = (
        "You are the Oracle in a GuessWhat?! game. You know exactly which object in the image is the target.\n\n"
        f"Candidate objects:\n{description}\n"
        f"Target:[{target_index}]\n"
        "Visible candidates are listed as [index] category with normalized bounding boxes.\n\n"
        "Answer every question about the target with exactly one word: Yes, No, or N/A. "
        "Do not explain your answer.\n"
        f"Question: {question}"
    )
    return {
        "images": [image],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
    }


def lora_config() -> dict:
    return {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }


def trainer_config() -> dict:
    return {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "eval_steps": 20,
        "max_length": 2048,
    }
