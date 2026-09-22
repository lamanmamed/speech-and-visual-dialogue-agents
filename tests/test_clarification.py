import unittest

from speech_visual_dialogue.clarification import (
    FOOD_MAP,
    build_clarification_response,
    decide_clarification,
    dm_preprocess,
    init_state,
    normalize_slot_verbose,
    parse_user_constraints,
    update_state_from_parsed,
)


class ClarificationTests(unittest.TestCase):
    def test_exact_and_fuzzy_mapping(self):
        exact = normalize_slot_verbose("veggie", FOOD_MAP)
        fuzzy = normalize_slot_verbose("chainese", FOOD_MAP)
        self.assertEqual(exact["value"], "vegetarian")
        self.assertEqual(exact["status"], "mapped")
        self.assertEqual(fuzzy["value"], "chinese")
        self.assertEqual(fuzzy["status"], "fuzzy")
        self.assertAlmostEqual(fuzzy["confidence"], 0.933, places=3)

    def test_unknown_is_different_from_missing(self):
        parsed = parse_user_constraints("I want asian food")
        self.assertEqual(parsed["food"]["status"], "unknown")
        self.assertTrue(parsed["food"]["mentioned"])
        self.assertEqual(parsed["area"]["status"], "missing")

    def test_fuzzy_value_is_not_committed(self):
        state = init_state()
        parsed = parse_user_constraints("I want chainese food")
        update_state_from_parsed(state, parsed)
        self.assertIsNone(state["constraints"]["food"])

    def test_medium_fuzzy_match_triggers_implicit_clarification(self):
        response, _, _ = dm_preprocess("I want chainese food", init_state())
        self.assertEqual(response["action"], "implicit_clarification")
        self.assertIn("chinese", response["response"])

    def test_oov_value_triggers_explicit_clarification(self):
        response, _, _ = dm_preprocess("I want asian food", init_state())
        self.assertEqual(response["action"], "explicit_clarification")
        self.assertIn("Did you mean one of these", response["response"])

    def test_missing_slot_triggers_follow_up(self):
        response, _, _ = dm_preprocess("I want Chinese food", init_state())
        self.assertEqual(response["action"], "follow_up")
        self.assertEqual(response["slot"], "area")

    def test_senter_falls_below_implicit_threshold(self):
        response, _, _ = dm_preprocess("Find me something in the senter", init_state())
        self.assertEqual(response["action"], "explicit_clarification")


if __name__ == "__main__":
    unittest.main()
