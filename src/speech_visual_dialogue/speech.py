"""Speech recognition and text-to-speech helpers used by the spoken dialogue pipeline."""

from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path


def transcribe_file(model, audio_path: str | Path, *, language: str = "en") -> str:
    """Transcribe one saved audio file with an already-loaded Whisper model."""
    result = model.transcribe(str(audio_path), language=language)
    return result["text"].strip()


def evaluate_whisper_model(
    model_name: str,
    audio_paths: list[str | Path],
    references: list[str],
    *,
    language: str = "en",
) -> dict[str, float]:
    """Evaluate one Whisper size with the same normalization used in the experiments."""
    if len(audio_paths) != len(references):
        raise ValueError("audio_paths and references must have the same length")

    try:
        import evaluate
        import whisper
        from whisper.normalizers import BasicTextNormalizer
    except ImportError as exc:
        raise ImportError("Install the project with the 'models' extra") from exc

    normalizer = BasicTextNormalizer()
    model = whisper.load_model(model_name)
    predictions = []
    start = time.time()

    for audio_path in audio_paths:
        prediction = transcribe_file(model, audio_path, language=language)
        predictions.append(normalizer(prediction))

    normalized_references = [normalizer(reference) for reference in references]
    wer = evaluate.load("wer").compute(
        predictions=predictions,
        references=normalized_references,
    )
    return {
        "wer": float(wer),
        "seconds": float(time.time() - start),
    }


def synthesize_gtts(text: str, *, language: str = "en") -> bytes:
    """Synthesize one response with gTTS and return the MP3 bytes."""
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise ImportError("Install the project with the 'models' extra") from exc

    buffer = BytesIO()
    gTTS(text=text, lang=language).write_to_fp(buffer)
    return buffer.getvalue()
