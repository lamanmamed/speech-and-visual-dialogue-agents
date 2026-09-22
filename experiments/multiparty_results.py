"""Recorded next-speaker results, including the fixed-seed ablation run."""

MAIN_RESULT = {
    "accuracy": 0.7158,
    "macro_precision": 0.7141,
    "macro_recall": 0.7141,
    "macro_f1": 0.7139,
}

ABLATIONS = [
    {"name": "full", "dev_f1": 0.7417, "test_f1": 0.7070, "test_accuracy": 0.7115},
    {"name": "no_current_speaker", "dev_f1": 0.7202, "test_f1": 0.6762, "test_accuracy": 0.6801},
    {"name": "no_history_speaker", "dev_f1": 0.3573, "test_f1": 0.3768, "test_accuracy": 0.3879},
    {"name": "text_only", "dev_f1": 0.2793, "test_f1": 0.2788, "test_accuracy": 0.2948},
]

if __name__ == "__main__":
    print("Main result:", MAIN_RESULT)
    for row in ABLATIONS:
        print(row)
