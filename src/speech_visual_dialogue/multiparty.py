"""Hierarchical BiLSTM model for predicting who speaks next in a four-person meeting."""

from __future__ import annotations

import re
from collections import Counter

import torch
from torch import nn

SPEAKER2ID = {"A": 0, "B": 1, "C": 2, "D": 3}


def simple_tokenize(text: str) -> list[str]:
    text = re.sub(r"[^\w\s']", " ", str(text).lower().strip())
    return re.sub(r"\s+", " ", text).split()


def build_vocab(texts: list[str], *, min_frequency: int = 2) -> dict[str, int]:
    counts = Counter(token for text in texts for token in simple_tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counts.items():
        if count >= min_frequency:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(token, vocab["<UNK>"]) for token in simple_tokenize(text)]
    return ids or [vocab["<UNK>"]]


def build_next_speaker_examples(turns: list[dict], *, history: int = 8) -> list[dict]:
    """Build fixed-history training examples from one meeting's ordered turns."""
    rows = []
    for index in range(history, len(turns) - 1):
        previous = turns[index - history:index]
        current = turns[index]
        next_turn = turns[index + 1]
        rows.append({
            "history_texts": [turn["text"] for turn in previous],
            "history_speakers": [SPEAKER2ID[turn["speaker"]] for turn in previous],
            "text": current["text"],
            "speaker": SPEAKER2ID[current["speaker"]],
            "label": SPEAKER2ID[next_turn["speaker"]],
        })
    return rows


class UtteranceBiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return self.dropout(torch.cat([hidden[0], hidden[1]], dim=1))


class NextSpeakerBiLSTMModel(nn.Module):
    """Encode utterances first, then model their order and speaker identities."""

    def __init__(
        self,
        vocab_size: int,
        *,
        history: int = 8,
        embedding_dim: int = 128,
        word_hidden: int = 128,
        context_hidden: int = 128,
        speaker_embedding_dim: int = 16,
        dropout: float = 0.3,
        use_current_speaker: bool = True,
        use_history_speaker: bool = True,
    ):
        super().__init__()
        self.history = history
        self.use_current_speaker = use_current_speaker
        self.use_history_speaker = use_history_speaker

        self.utterance_encoder = UtteranceBiLSTMEncoder(
            vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=word_hidden,
            dropout=dropout,
        )
        utterance_dim = 2 * word_hidden

        if use_history_speaker:
            self.history_speaker_embedding = nn.Embedding(4, speaker_embedding_dim)
            context_input = utterance_dim + speaker_embedding_dim
        else:
            self.history_speaker_embedding = None
            context_input = utterance_dim

        self.context_lstm = nn.LSTM(
            context_input,
            context_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.current_speaker_embedding = nn.Embedding(4, speaker_embedding_dim)
        self.dropout = nn.Dropout(dropout)

        classifier_input = 2 * context_hidden
        if use_current_speaker:
            classifier_input += speaker_embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, classifier_input),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input, 4),
        )

    def forward(
        self,
        history_tokens: list[torch.Tensor],
        history_lengths: list[torch.Tensor],
        history_speakers: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        speaker: torch.Tensor,
    ) -> torch.Tensor:
        history_vectors = []
        batch_size = tokens.shape[0]

        for index, (token_batch, length_batch) in enumerate(zip(history_tokens, history_lengths)):
            vector = self.utterance_encoder(token_batch, length_batch)
            if self.use_history_speaker:
                speaker_vector = self.history_speaker_embedding(history_speakers[:, index])
                vector = torch.cat([vector, speaker_vector], dim=1)
            history_vectors.append(vector)

        current = self.utterance_encoder(tokens, lengths)
        if self.use_history_speaker:
            zeros = torch.zeros(
                batch_size,
                self.history_speaker_embedding.embedding_dim,
                device=tokens.device,
            )
            current = torch.cat([current, zeros], dim=1)

        sequence = torch.stack(history_vectors + [current], dim=1)
        _, (hidden, _) = self.context_lstm(sequence)
        context = self.dropout(torch.cat([hidden[0], hidden[1]], dim=1))

        if self.use_current_speaker:
            current_speaker = self.current_speaker_embedding(speaker)
            context = torch.cat([context, current_speaker], dim=1)

        return self.classifier(context)
