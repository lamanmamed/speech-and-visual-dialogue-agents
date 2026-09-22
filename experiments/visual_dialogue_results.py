"""Recorded 20-game GuessWhat?! comparison."""

RESULTS = {
    "zero_shot": {
        "target_selection_accuracy": 0.20,
        "oracle_strict_valid_rate": 1.00,
        "oracle_recoverable_rate": 1.00,
    },
    "lora_oracle": {
        "target_selection_accuracy": 0.25,
        "oracle_strict_valid_rate": 0.99,
        "oracle_recoverable_rate": 1.00,
    },
}

if __name__ == "__main__":
    for name, metrics in RESULTS.items():
        print(name, metrics)
