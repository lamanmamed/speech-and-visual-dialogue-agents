"""Deterministic clarification policy for noisy restaurant requests."""

from __future__ import annotations

import difflib
from typing import Any

AREA_MAP = {
    "centre": "centre",
    "center": "centre",
    "city centre": "centre",
    "city center": "centre",
    "downtown": "centre",
    "central": "centre",
    "north": "north",
    "south": "south",
    "east": "east",
    "west": "west",
    "riverside": "riverside",
    "river side": "riverside",
    "riverside area": "riverside",
}

FOOD_MAP = {
    "chinese": "chinese",
    "italian": "italian",
    "indian": "indian",
    "japanese": "japanese",
    "vegetarian": "vegetarian",
    "veggie": "vegetarian",
    "british": "british",
    "english": "british",
}

PRICE_MAP = {
    "cheap": "cheap",
    "inexpensive": "cheap",
    "budget": "cheap",
    "low": "cheap",
    "low priced": "cheap",
    "moderate": "moderate",
    "medium": "moderate",
    "moderately priced": "moderate",
    "expensive": "expensive",
    "pricey": "expensive",
    "high": "expensive",
    "high end": "expensive",
}

AREA_HINTS = [
    "area", "location", "part of town", "part", "side", "district",
    "centre", "center", "central", "north", "south", "east", "west",
    "riverside", "river side", "downtown",
]
FOOD_HINTS = [
    "food", "cuisine", "eat", "meal", "chinese", "italian", "indian",
    "japanese", "vegetarian", "veggie", "british", "english", "asian", "oriental",
]
PRICE_HINTS = [
    "price", "priced", "cost", "cheap", "budget", "expensive", "pricey",
    "moderate", "mid", "medium", "midscale", "premium", "high end",
]


def normalize_slot_verbose(value, mapping: dict[str, str], fuzzy_cutoff: float = 0.72) -> dict[str, Any]:
    """Map exact, substring, and fuzzy slot values while keeping unknown separate from missing."""
    raw = value
    if value is None or not str(value).strip():
        return {"raw": raw, "value": None, "status": "missing", "confidence": 0.0, "candidates": []}

    text = str(value).strip().lower()
    if text in mapping:
        canonical = mapping[text]
        return {"raw": raw, "value": canonical, "status": "mapped", "confidence": 1.0, "candidates": [canonical]}

    substring_hits = [key for key in mapping if key in text]
    if substring_hits:
        best_key = max(substring_hits, key=len)
        canonical = mapping[best_key]
        return {"raw": raw, "value": canonical, "status": "substring", "confidence": 0.95, "candidates": [canonical]}

    tokens = text.split()
    spans = list(tokens)
    spans.extend(f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1))
    spans.append(text)

    best_ratio = 0.0
    best_key = None
    for span in spans:
        for key in mapping:
            ratio = difflib.SequenceMatcher(None, span, key).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key

    if best_key is not None and best_ratio >= fuzzy_cutoff:
        canonical = mapping[best_key]
        return {
            "raw": raw,
            "value": canonical,
            "status": "fuzzy",
            "confidence": round(best_ratio, 3),
            "candidates": [canonical],
        }

    return {"raw": raw, "value": None, "status": "unknown", "confidence": 0.0, "candidates": []}


def _detect_slot_mention(text: str, hints: list[str]) -> bool:
    return any(hint in text for hint in hints)


def _parse_slot(text: str, mapping: dict[str, str], hints: list[str]) -> dict[str, Any]:
    info = normalize_slot_verbose(text, mapping)
    if info["status"] == "unknown":
        if _detect_slot_mention(text, hints):
            info["mentioned"] = True
        else:
            info["status"] = "missing"
            info["mentioned"] = False
    else:
        info["mentioned"] = info["status"] != "missing"
    return info


def parse_user_constraints(user_text: str) -> dict[str, dict[str, Any]]:
    text = user_text.lower()
    return {
        "area": _parse_slot(text, AREA_MAP, AREA_HINTS),
        "food": _parse_slot(text, FOOD_MAP, FOOD_HINTS),
        "pricerange": _parse_slot(text, PRICE_MAP, PRICE_HINTS),
    }


def init_state() -> dict[str, Any]:
    return {
        "constraints": {"area": None, "food": None, "pricerange": None, "name": None},
        "booking": {
            "restaurant_id": None,
            "day": None,
            "time": None,
            "people": None,
            "customer_name": None,
        },
        "pending_clarification": None,
        "last_results": [],
    }


def update_state_from_parsed(state: dict[str, Any], parsed_slots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Commit only mapped or substring values. Fuzzy and unknown values stay provisional."""
    for slot in ("area", "food", "pricerange"):
        info = parsed_slots[slot]
        if info["status"] in {"mapped", "substring"}:
            state["constraints"][slot] = info["value"]
    return state


def get_next_missing_slot(state: dict[str, Any], exclude_slot: str | None = None) -> str | None:
    for slot in ("area", "food", "pricerange"):
        if slot != exclude_slot and state["constraints"][slot] is None:
            return slot
    return None


def decide_clarification(
    state: dict[str, Any],
    parsed_slots: dict[str, dict[str, Any]],
    *,
    low_conf: float = 0.84,
    medium_conf: float = 0.94,
) -> dict[str, Any] | None:
    """Choose explicit clarification, implicit clarification, follow-up, or no intervention."""
    for slot in ("area", "food", "pricerange"):
        info = parsed_slots[slot]
        if info["status"] == "unknown" and info.get("mentioned"):
            return {
                "action": "explicit_clarification",
                "slot": slot,
                "reason": "oov_value",
                "value": None,
                "confidence": info["confidence"],
            }
        if info["status"] == "fuzzy" and info["confidence"] < low_conf:
            return {
                "action": "explicit_clarification",
                "slot": slot,
                "reason": "low_confidence_fuzzy",
                "value": info["value"],
                "confidence": info["confidence"],
            }

    for slot in ("area", "food", "pricerange"):
        info = parsed_slots[slot]
        if info["status"] == "fuzzy" and low_conf <= info["confidence"] < medium_conf:
            return {
                "action": "implicit_clarification",
                "slot": slot,
                "value": info["value"],
                "confidence": info["confidence"],
                "next_slot": get_next_missing_slot(state, exclude_slot=slot),
            }

    next_missing = get_next_missing_slot(state)
    if next_missing is not None:
        return {"action": "follow_up", "slot": next_missing}
    return None


def _slot_values_text(slot: str) -> str:
    return {
        "area": "centre, north, south, east, west, or riverside",
        "food": "chinese, italian, indian, japanese, vegetarian, or british",
        "pricerange": "cheap, moderate, or expensive",
    }.get(slot, "")


def _follow_up_question(slot: str) -> str:
    return {
        "area": "What area are you looking for?",
        "food": "What kind of food would you like?",
        "pricerange": "What price range would you like?",
    }.get(slot, "Could you tell me a bit more?")


def build_clarification_response(decision: dict[str, Any]) -> dict[str, Any]:
    action = decision["action"]
    slot = decision.get("slot")

    if action == "follow_up":
        return {"action": action, "slot": slot, "response": _follow_up_question(slot)}

    if action == "explicit_clarification":
        if decision.get("reason") == "oov_value":
            response = f"I didn't catch the {slot} clearly. Did you mean one of these: {_slot_values_text(slot)}?"
        else:
            response = f"I think you said {decision.get('value')}. Is that right?"
        return {**decision, "response": response}

    if action == "implicit_clarification":
        value = decision.get("value")
        next_slot = decision.get("next_slot")
        if next_slot is None:
            response = f"OK, I'll look for a {value} {slot}."
        elif next_slot == "area":
            response = f"OK, in what area are you looking for a {value} restaurant?"
        elif next_slot == "food":
            response = f"OK, what kind of food would you like in the {value} area?"
        elif next_slot == "pricerange":
            response = f"OK, what price range would you like for a {value} restaurant?"
        else:
            response = f"OK, could you tell me more about your {next_slot}?"
        return {**decision, "response": response}

    return {"action": "final", "response": "Could you tell me a bit more about what you're looking for?"}


def dm_preprocess(user_text: str, state: dict[str, Any]):
    """Run slot parsing and clarification before a generative agent sees the turn."""
    parsed = parse_user_constraints(user_text)
    state = update_state_from_parsed(state, parsed)
    decision = decide_clarification(state, parsed)

    if decision is None:
        state["pending_clarification"] = None
        return None, state, parsed

    response = build_clarification_response(decision)
    state["pending_clarification"] = {
        "action": decision["action"],
        "slot": decision.get("slot"),
        "value": decision.get("value"),
    }
    return response, state, parsed
