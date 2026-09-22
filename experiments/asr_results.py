"""Recorded Whisper results from the spoken-language experiments."""

FLEURS_RESULTS = {
    "tiny": {"wer": 0.1538, "seconds": 188.53},
    "base": {"wer": 0.1071, "seconds": 196.05},
    "small": {"wer": 0.0696, "seconds": 291.42},
    "medium": {"wer": 0.0513, "seconds": 471.30},
}

RECORDED_TRANSCRIPTS = {
    "whispered_alzheimers": {
        "tiny": "pr muit",
        "base": "Who is Alzheimer's?",
        "small": "Always Alzheimer's",
        "medium": "Who is Alzheimer's?",
    },
    "music_barbican": {
        "tiny": "What is barbecue?",
        "base": "What is barbeque?",
        "small": "What is Barbie?",
        "medium": "What is Barbican?",
    },
    "clean_barbican": {
        "tiny": "What is barbeque?",
        "base": "What is Barbican?",
        "small": "What is Barbican?",
        "medium": "What is Barbican?",
    },
    "sung_question": {
        "tiny": "Do you say, I have forgotten about you?",
        "base": "Do you think I have forgotten about you?",
        "small": "Do you think I have forgotten about you?",
        "medium": "Do you think I have forgotten about you?",
    },
    "crosstalk_distance": {
        "base": "I mean the kilometers is one and more.",
        "small": "Our main kilometers is one mile.",
        "medium": "How many kilometers is one mile?",
    },
    "clean_distance": {
        "tiny": "How many kilometers is one mile?",
        "base": "How many kilometers is one mile?",
        "small": "How many kilometers is one mile?",
        "medium": "How many kilometers is one mile?",
    },
}

if __name__ == "__main__":
    for model, result in FLEURS_RESULTS.items():
        print(f"{model:>6}: WER={result['wer']:.2%}, time={result['seconds']:.2f}s")
