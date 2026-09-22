import unittest

from speech_visual_dialogue.visual_dialogue import (
    build_candidate_description,
    build_oracle_text,
    build_questioner_text,
    extract_candidate_index,
    normalize_oracle_answer,
    normalized_bbox,
    oracle_output_metrics,
)


CANDIDATES = [
    {"category": "person", "bbox": [10, 20, 30, 40]},
    {"category": "kite", "bbox": [60, 10, 20, 20]},
]


class VisualDialogueTests(unittest.TestCase):
    def test_bbox_is_normalized_to_qwen_scale(self):
        box = normalized_bbox(CANDIDATES[0], (100, 100))
        self.assertEqual(box, [100, 200, 400, 599])

    def test_candidate_description_contains_indices(self):
        text = build_candidate_description(CANDIDATES, (100, 100))
        self.assertIn("[0] person", text)
        self.assertIn("[1] kite", text)

    def test_questioner_prompt_tracks_elimination(self):
        text = build_questioner_text(
            CANDIDATES,
            (100, 100),
            [("Is the target candidate [0] a person?", "No")],
        )
        self.assertIn("Eliminated so far: ['0']", text)
        self.assertIn("Still possible: ['1']", text)

    def test_oracle_prompt_contains_hidden_index(self):
        text = build_oracle_text(CANDIDATES, (100, 100), 1, "Is it a kite?")
        self.assertIn("Index: 1", text)
        self.assertIn("Category: kite", text)

    def test_oracle_normalization_recovers_extra_text(self):
        self.assertEqual(normalize_oracle_answer("Yes, because..."), "Yes")
        self.assertEqual(normalize_oracle_answer("No."), "No")
        self.assertIsNone(normalize_oracle_answer("Maybe"))

    def test_oracle_output_metrics(self):
        metrics = oracle_output_metrics(["Yes", "No.", "N/A"])
        self.assertAlmostEqual(metrics["strict_valid_rate"], 2 / 3)
        self.assertEqual(metrics["recoverable_rate"], 1.0)

    def test_selector_index_extraction(self):
        self.assertEqual(extract_candidate_index("The answer is [3]"), 3)
        self.assertIsNone(extract_candidate_index("candidate three"))


if __name__ == "__main__":
    unittest.main()
