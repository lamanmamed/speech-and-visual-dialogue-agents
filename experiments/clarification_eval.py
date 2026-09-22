"""Reproduce the 12-case clarification-policy evaluation."""

from speech_visual_dialogue.clarification import dm_preprocess, init_state

CASES = [
    ("M1", "I want Chinese food.", "follow_up"),
    ("M2", "I want a cheap place.", "follow_up"),
    ("M3", "Find me somewhere in the centre.", "follow_up"),
    ("A1", "I want chainese food.", "implicit_clarification"),
    ("A2", "Find me something in the senter.", "implicit_clarification"),
    ("O1", "I want asian food.", "explicit_clarification"),
    ("O2", "Find me a midscale place.", "explicit_clarification"),
    ("O3", "I'm looking for oriental food in the west.", "explicit_clarification"),
    ("N1", "i want a cheap chainese place", "implicit_clarification"),
    ("N2", "modrate japanees food", "implicit_clarification"),
    ("N3", "i want a restarant in the senter", "explicit_clarification"),
    ("N4", "book for to people at seven", "follow_up"),
]

correct = 0
for case_id, utterance, expected in CASES:
    response, _, _ = dm_preprocess(utterance, init_state())
    predicted = response["action"] if response else "none"
    matched = predicted == expected
    correct += int(matched)
    print(case_id, expected, predicted, "✓" if matched else "✗")

print(f"Accuracy: {correct}/{len(CASES)} = {correct / len(CASES):.3f}")
