import unittest

from tts_service.models import QwenEngine


class FakeModel:
    def __init__(self):
        self.calls = []

    def get_supported_speakers(self):
        return ["Vivian"]

    def get_supported_languages(self):
        return ["Chinese"]

    def generate_custom_voice(self, **kw):
        self.calls.append(("custom", kw))
        return [0.1, 0.2], 24000

    def generate_voice_design(self, **kw):
        self.calls.append(("design", kw))
        return [0.1], 24000

    def generate_voice_clone(self, **kw):
        self.calls.append(("clone", kw))
        return [0.1], 24000


class QwenEngineTests(unittest.TestCase):
    def engine(self):
        e = QwenEngine("c", "d", "b")
        e._custom = FakeModel()
        e._design = FakeModel()
        e._clone = FakeModel()
        return e

    def test_speak_custom_passes_speaker_and_instruct(self):
        e = self.engine()
        samples, sr = e.speak_custom("你好", "Chinese", "Vivian", "温柔")
        kind, kw = e._custom.calls[0]
        self.assertEqual((kind, kw["text"], kw["speaker"], kw["instruct"]),
                         ("custom", "你好", "Vivian", "温柔"))
        self.assertEqual(sr, 24000)
        self.assertEqual(list(samples), [0.1, 0.2])

    def test_speak_design_requires_instruct(self):
        e = self.engine()
        with self.assertRaises(ValueError):
            e.speak_design("你好", "Chinese", "")
        e.speak_design("你好", "Chinese", "年轻女性")
        self.assertEqual(e._design.calls[0][0], "design")

    def test_speak_clone_passes_ref_audio(self):
        e = self.engine()
        e.speak_clone("你好", "Chinese", "/tmp/a.wav", "参考文本", False)
        kind, kw = e._clone.calls[0]
        self.assertEqual((kind, kw["ref_audio"], kw["ref_text"]),
                         ("clone", "/tmp/a.wav", "参考文本"))


if __name__ == "__main__":
    unittest.main()
