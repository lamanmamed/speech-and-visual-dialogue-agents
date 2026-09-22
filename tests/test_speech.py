import unittest
from types import SimpleNamespace

from speech_visual_dialogue.speech import transcribe_file


class DummyWhisper:
    def transcribe(self, path, language="en"):
        self.path = path
        self.language = language
        return {"text": "  hello world  "}


class SpeechTests(unittest.TestCase):
    def test_transcribe_file_returns_stripped_text(self):
        model = DummyWhisper()
        self.assertEqual(transcribe_file(model, "sample.wav"), "hello world")
        self.assertEqual(model.language, "en")


if __name__ == "__main__":
    unittest.main()
