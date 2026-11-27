import os
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="imagehub-new-models-")
os.environ.setdefault("IMAGEHUB_DB", os.path.join(TMP, "imagehub.db"))
os.environ.setdefault("IMAGEHUB_THUMBS", os.path.join(TMP, "thumbs"))
os.environ.setdefault("IMAGEHUB_RUN", os.path.join(TMP, "run"))
os.environ.setdefault("IMAGEHUB_OUTPUT", os.path.join(TMP, "output"))

import app as imagehub  # noqa: E402


class NewModelTests(unittest.TestCase):
    def test_flux_workflow(self):
        wf = imagehub.WF_BUILDERS["flux2klein"]("a cat", 1024, 768, 42)
        classes = {v["class_type"] for v in wf.values()}
        self.assertTrue({"UNETLoader", "CLIPLoader", "VAELoader",
                         "Flux2Scheduler", "SamplerCustomAdvanced",
                         "VAEDecode", "SaveImage"} <= classes)
        unet = next(v for v in wf.values() if v["class_type"] == "UNETLoader")
        self.assertEqual(unet["inputs"]["unet_name"],
                         "flux-2-klein-4b-fp8.safetensors")
        sched = next(v for v in wf.values() if v["class_type"] == "Flux2Scheduler")
        self.assertEqual(sched["inputs"]["steps"], 4)
        self.assertEqual(sched["inputs"]["width"], 1024)

    def test_nemotron_workflow(self):
        wf = imagehub.WF_BUILDERS["nemotron"]("a cat", 1024, 1024, 42)
        classes = {v["class_type"] for v in wf.values()}
        self.assertTrue({"NemotronImageLoader", "NemotronImageGenerate",
                         "NemotronImageDecoderLoader", "VAEDecode",
                         "SaveImage"} <= classes)
        loader = next(v for v in wf.values()
                      if v["class_type"] == "NemotronImageLoader")
        self.assertIn("w4a8.safetensors", loader["inputs"]["model_name"])
        gen = next(v for v in wf.values()
                   if v["class_type"] == "NemotronImageGenerate")
        self.assertEqual(gen["inputs"]["width"], 1024)
        self.assertEqual(gen["inputs"]["seed"], 42)
        dec = next(v for v in wf.values()
                   if v["class_type"] == "NemotronImageDecoderLoader")
        self.assertIn("tokenizer_bf16", dec["inputs"]["vae_name"])

    def test_model_ready_comfy_only(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "w.safetensors")
        open(path, "w").close()
        with mock.patch.dict(imagehub.MODELS["flux2klein"], {"needs": [path]}):
            self.assertTrue(imagehub._model_ready("flux2klein"))
            os.remove(path)
            self.assertFalse(imagehub._model_ready("flux2klein"))

    def test_comfy_only_forces_comfy_backend(self):
        calls = []

        def fake_comfy(model_key, prompt, w, h, seed):
            calls.append((model_key, prompt, w, h, seed))
            return {"model": model_key, "status": "ok", "image": "x.png"}

        with mock.patch.object(imagehub, "_run_comfy", side_effect=fake_comfy):
            res = imagehub._run_model_impl("nemotron", "a cat", 1024, 1024,
                                           42, backend="direct")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(calls, [("nemotron", "a cat", 1024, 1024, 42)])

    def test_aliases(self):
        self.assertEqual(imagehub.MODEL_ALIASES["flux"], "flux2klein")
        self.assertEqual(imagehub.MODEL_ALIASES["flux-2-klein"], "flux2klein")
        self.assertEqual(imagehub.MODEL_ALIASES["nemotron"], "nemotron")


if __name__ == "__main__":
    unittest.main()
