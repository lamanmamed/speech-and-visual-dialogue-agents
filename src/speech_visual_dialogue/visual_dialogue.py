"""Prompt and output handling for a two-agent GuessWhat?! visual dialogue system."""

from __future__ import annotations

import re
from typing import Optional

from PIL import Image, ImageDraw

VALID_ORACLE_ANSWERS = {"yes", "no", "n/a"}


def _to_qwen_coordinate(value: float, size: int) -> int:
    if size <= 1:
        return 0
    return round(value / size * 999)


def normalized_bbox(obj: dict, image_size: tuple[int, int]) -> list[int]:
    """Convert a COCO [x, y, width, height] box to Qwen's 0-999 coordinate scale."""
    width, height = image_size
    x, y, box_width, box_height = obj["bbox"]
    return [
        _to_qwen_coordinate(x, width),
        _to_qwen_coordinate(y, height),
        _to_qwen_coordinate(x + box_width, width),
        _to_qwen_coordinate(y + box_height, height),
    ]


def build_candidate_description(candidates: list[dict], image_size: tuple[int, int]) -> str:
    lines = []
    for index, obj in enumerate(candidates):
        box = normalized_bbox(obj, image_size)
        lines.append(f'[{index}] {obj["category"]} {{"bbox_2d": {box}}}')
    return "\n".join(lines)


def draw_candidates(image: Image.Image, candidates: list[dict]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for index, obj in enumerate(candidates):
        x, y, width, height = obj["bbox"]
        draw.rectangle((x, y, x + width, y + height), outline="white", width=2)
        draw.text((x + 2, y + 2), str(index), fill="white")
    return output


def _history_lines(dialogue_history: list) -> tuple[str, list[str]]:
    lines = []
    eliminated = []
    for turn in dialogue_history:
        if isinstance(turn, dict):
            question, answer = turn.get("question", ""), turn.get("answer", "")
        elif isinstance(turn, (tuple, list)) and len(turn) >= 2:
            question, answer = turn[0], turn[1]
        else:
            question, answer = str(turn), ""
        lines.append(f"Q: {question}\nA: {answer}")
        indices = re.findall(r"\[(\d+)\]", question)
        if str(answer).lower() == "no" and indices:
            eliminated.extend(indices)
    return "\n".join(lines) if lines else "None", eliminated


def build_questioner_text(candidates: list[dict], image_size: tuple[int, int], dialogue_history: list) -> str:
    description = build_candidate_description(candidates, image_size)
    history, eliminated = _history_lines(dialogue_history)
    remaining = [str(i) for i in range(len(candidates)) if str(i) not in eliminated]
    elimination_note = ""
    if dialogue_history:
        elimination_note = f"\nEliminated so far: {eliminated}\nStill possible: {remaining}"

    return f"""You are the Questioner in a visual guessing game. A hidden target object is in the image. Your goal is to identify it by asking yes/no questions.

Candidates:
{description}

Previous Q&A:
{history}{elimination_note}

Rules:
- Ask ONE yes/no question only.
- Use only candidate indices from the list above.
- Prefer questions that name a specific candidate index and its category.
- Do NOT repeat a question already asked above.
- Do NOT ask about eliminated candidates.
- Output ONLY the question, starting with "Is".

Your question:"""


def build_oracle_text(
    candidates: list[dict],
    image_size: tuple[int, int],
    target_index: int,
    question: str,
) -> str:
    description = build_candidate_description(candidates, image_size)
    target = candidates[target_index]
    return f"""You are the Oracle in a visual target-guessing game.

Candidate objects:
{description}

Hidden target:
Index: {target_index}
Category: {target["category"]}
BBox: {target["bbox"]}

Question:
{question}

Answer using exactly one token: Yes, No, or N/A.

Decision rules:
- If the question asks whether candidate [k] is the target, answer Yes only if k equals the hidden target index.
- If k is not the hidden target index, answer No, even if candidate [k] has the correct category.
- If the question asks about the target category, compare it with the hidden target category.
- If the question is not answerable as yes/no, answer N/A.
- Do not explain the answer."""


def build_selector_text(dialogue_history: list, candidates: list[dict], image_size: tuple[int, int]) -> str:
    history = "\n".join(f"{role}: {text.strip()}" for role, text in dialogue_history) if dialogue_history else "None"
    description = build_candidate_description(candidates, image_size)
    return f"""Dialogue history:
{history}

Candidates:
{description}

Select the single target candidate consistent with the dialogue.
Return only one index in square brackets, for example [0]."""


def normalize_oracle_answer(raw: str) -> Optional[str]:
    cleaned = raw.strip().lower()
    if cleaned in VALID_ORACLE_ANSWERS:
        return cleaned.upper() if cleaned == "n/a" else cleaned.capitalize()
    for token in ("yes", "no", "n/a"):
        if cleaned.startswith(token):
            return token.upper() if token == "n/a" else token.capitalize()
    match = re.search(r"\b(yes|no|n/a)\b", cleaned)
    if match:
        token = match.group(1)
        return token.upper() if token == "n/a" else token.capitalize()
    return None


def is_strict_oracle_answer(raw: str) -> bool:
    return raw.strip().lower() in VALID_ORACLE_ANSWERS


def oracle_output_metrics(outputs: list[str]) -> dict[str, float]:
    if not outputs:
        raise ValueError("outputs cannot be empty")
    strict = sum(is_strict_oracle_answer(output) for output in outputs) / len(outputs)
    recoverable = sum(normalize_oracle_answer(output) is not None for output in outputs) / len(outputs)
    return {"n": len(outputs), "strict_valid_rate": strict, "recoverable_rate": recoverable}


def extract_candidate_index(text: str) -> int | None:
    matches = re.findall(r"\[(\d+)\]", text)
    return int(matches[-1]) if matches else None
