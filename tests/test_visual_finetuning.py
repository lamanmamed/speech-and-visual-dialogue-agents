import unittest

from PIL import Image

from speech_visual_dialogue.visual_finetuning import (
    build_oracle_training_example,
    lora_config,
    trainer_config,
)


class VisualFineTuningTests(unittest.TestCase):
    def test_lora_configuration_matches_recorded_run(self):
        config = lora_config()
        self.assertEqual(config["r"], 16)
        self.assertEqual(config["lora_alpha"], 32)
        self.assertEqual(config["target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"])

    def test_trainer_configuration_matches_recorded_run(self):
        config = trainer_config()
        self.assertEqual(config["num_train_epochs"], 2)
        self.assertEqual(config["gradient_accumulation_steps"], 4)
        self.assertEqual(config["learning_rate"], 2e-4)

    def test_training_example_contains_image_and_answer(self):
        image = Image.new("RGB", (100, 80))
        candidates = [{"category": "cat", "bbox": [10, 10, 20, 20]}]
        example = build_oracle_training_example(image, "Is it a cat?", "Yes", candidates, 0)
        self.assertEqual(example["images"][0].size, (100, 80))
        self.assertEqual(example["messages"][1]["content"][0]["text"], "Yes")


if __name__ == "__main__":
    unittest.main()
