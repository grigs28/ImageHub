import io
import os
import tempfile
import unittest
import wave

from tts_service import audio


def make_wav(path, seconds=1.0, sr=16000, channels=2):
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        n = int(seconds * sr)
        w.writeframes(b"\x00\x00" * channels * n)


class AudioTests(unittest.TestCase):
    def test_wav_bytes_roundtrip(self):
        data = audio.wav_bytes([0.0, 0.5, -0.5], 24000)
        with wave.open(io.BytesIO(data)) as w:
            self.assertEqual((w.getnchannels(), w.getframerate()), (1, 24000))
            self.assertEqual(w.getnframes(), 3)

    def test_convert_to_24k_mono(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "in.wav")
        dst = os.path.join(d, "out.wav")
        make_wav(src, 1.0, 16000, 2)
        audio.convert_to_wav(src, dst)
        with wave.open(dst) as w:
            self.assertEqual((w.getnchannels(), w.getframerate()), (1, 24000))
            self.assertAlmostEqual(w.getnframes() / 24000, 1.0, places=1)

    def test_convert_rejects_long_audio(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "long.wav")
        dst = os.path.join(d, "out.wav")
        make_wav(src, 61.0, 1000, 1)
        with self.assertRaises(ValueError):
            audio.convert_to_wav(src, dst)
        self.assertFalse(os.path.exists(dst))

    def test_convert_rejects_large_file(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "big.wav")
        with open(src, "wb") as f:
            f.write(b"\0" * (20 * 1024 * 1024 + 1))
        with self.assertRaises(ValueError):
            audio.convert_to_wav(src, os.path.join(d, "out.wav"))


if __name__ == "__main__":
    unittest.main()
