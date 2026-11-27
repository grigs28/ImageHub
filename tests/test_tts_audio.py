import io
import os
import tempfile
import unittest
import wave
from unittest import mock

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

    def test_convert_hardens_ffmpeg(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "in.bin")
        dst = os.path.join(d, "out.wav")
        with open(src, "wb") as f:
            f.write(b"not really audio")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append({"cmd": cmd, "kwargs": kwargs})
            make_wav(cmd[-1], 1.0, 24000, 1)
            return mock.Mock(returncode=0)

        with mock.patch.object(audio, "_probe_audio", return_value=("wav", 1.0)), \
                mock.patch.object(audio.subprocess, "run", side_effect=fake_run):
            audio.convert_to_wav(src, dst)

        cmd = calls[0]["cmd"]
        self.assertIn("-nostdin", cmd)
        self.assertIn("-protocol_whitelist", cmd)
        self.assertIn("file", cmd)
        self.assertIn("-t", cmd)

    def test_validate_rejects_playlist_format(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "in.wav")
        with open(src, "wb") as f:
            f.write(b"ffconcat version 1.0\nfile '/etc/passwd'\n")
        with mock.patch.object(audio, "_probe_audio", return_value=("ffconcat", 1.0)):
            with self.assertRaises(ValueError):
                audio.validate_ref_audio(src)

    def test_validate_rejects_long_probed_audio(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "in.mp3")
        with open(src, "wb") as f:
            f.write(b"tiny")
        with mock.patch.object(audio, "_probe_audio", return_value=("mp3", 61.0)):
            with self.assertRaises(ValueError):
                audio.validate_ref_audio(src)


if __name__ == "__main__":
    unittest.main()
