import unittest

import torch

from speech_visual_dialogue.multiparty import (
    NextSpeakerBiLSTMModel,
    build_next_speaker_examples,
    build_vocab,
    encode_text,
)


class MultiPartyTests(unittest.TestCase):
    def test_vocab_and_encoding(self):
        vocab = build_vocab(["hello there", "hello again"], min_frequency=1)
        self.assertIn("hello", vocab)
        self.assertEqual(encode_text("unknown-token", vocab), [vocab["<UNK>"], vocab["<UNK>"]])

    def test_build_examples_uses_eight_turn_history(self):
        turns = [
            {"speaker": "ABCD"[i % 4], "text": f"turn {i}"}
            for i in range(12)
        ]
        rows = build_next_speaker_examples(turns, history=8)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(rows[0]["history_texts"]), 8)
        self.assertEqual(rows[0]["label"], 1)

    def test_full_model_output_shape(self):
        batch = 3
        history = 8
        model = NextSpeakerBiLSTMModel(
            100,
            history=history,
            embedding_dim=16,
            word_hidden=12,
            context_hidden=10,
            speaker_embedding_dim=4,
            dropout=0.0,
        )
        history_tokens = [torch.randint(1, 100, (batch, 6)) for _ in range(history)]
        history_lengths = [torch.full((batch,), 6) for _ in range(history)]
        history_speakers = torch.randint(0, 4, (batch, history))
        tokens = torch.randint(1, 100, (batch, 7))
        lengths = torch.full((batch,), 7)
        speaker = torch.randint(0, 4, (batch,))
        logits = model(history_tokens, history_lengths, history_speakers, tokens, lengths, speaker)
        self.assertEqual(logits.shape, (batch, 4))

    def test_text_only_ablation_runs(self):
        batch = 2
        history = 2
        model = NextSpeakerBiLSTMModel(
            30,
            history=history,
            embedding_dim=8,
            word_hidden=6,
            context_hidden=5,
            speaker_embedding_dim=3,
            dropout=0.0,
            use_current_speaker=False,
            use_history_speaker=False,
        )
        history_tokens = [torch.randint(1, 30, (batch, 4)) for _ in range(history)]
        history_lengths = [torch.full((batch,), 4) for _ in range(history)]
        history_speakers = torch.randint(0, 4, (batch, history))
        tokens = torch.randint(1, 30, (batch, 5))
        lengths = torch.full((batch,), 5)
        speaker = torch.randint(0, 4, (batch,))
        self.assertEqual(
            model(history_tokens, history_lengths, history_speakers, tokens, lengths, speaker).shape,
            (batch, 4),
        )


if __name__ == "__main__":
    unittest.main()
