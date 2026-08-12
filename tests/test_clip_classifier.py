import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from experiments.activation_steering.clip_classifier import (
    ZeroShotCLIPConceptScorer,
)


class _FakeCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def encode_text(self, tokens):
        return tokens.float()

    def encode_image(self, images):
        return images.mean(dim=(-1, -2))[:, :2]


class _RecordingPreprocess:
    def __init__(self):
        self.modes = []

    def __call__(self, image):
        self.modes.append(image.mode)
        pixels = torch.tensor(bytearray(image.tobytes()), dtype=torch.float32)
        pixels = pixels.reshape(image.height, image.width, 3).permute(2, 0, 1)
        return pixels / 255.0


class CLIPConceptScorerTest(unittest.TestCase):
    def setUp(self):
        self.prompt_vectors = {
            "positive one": [0.0, 1.0],
            "positive two": [0.2, 1.0],
        }
        self.prompts = {
            "attribute": ["positive one", "positive two"],
        }
        self.preprocess = _RecordingPreprocess()
        self.scorer = ZeroShotCLIPConceptScorer(
            model_name="fake",
            pretrained="fake",
            prompts=self.prompts,
            device="cpu",
            model=_FakeCLIP(),
            preprocess=self.preprocess,
            tokenizer=self._tokenize,
        )

    def _tokenize(self, prompts):
        return torch.tensor([self.prompt_vectors[prompt] for prompt in prompts])

    def _save_image(self, path, mode, color):
        Image.new(mode, (2, 2), color=color).save(path)

    def test_positive_prompt_ensemble_is_normalized(self):
        prototype = self.scorer.text_prototypes["attribute"]

        torch.testing.assert_close(
            prototype.norm(), torch.tensor(1.0), atol=1e-6, rtol=1e-6
        )
        self.assertGreater(prototype[1], prototype[0])

    def test_predict_scores_preserves_batch_order_and_cosine_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [Path(tmpdir) / name for name in ("red.png", "green.png")]
            self._save_image(paths[0], "RGB", (255, 0, 0))
            self._save_image(paths[1], "RGB", (0, 255, 0))

            scores = self.scorer.predict_scores(paths, batch_size=1)["attribute"]

        self.assertEqual(scores.shape, (2,))
        self.assertTrue(torch.all((-1.0 <= scores) & (scores <= 1.0)))
        self.assertLess(scores[0].item(), scores[1].item())

    def test_score_paths_returns_similarities_and_converts_images_to_rgb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [Path(tmpdir) / name for name in ("gray.png", "rgba.png")]
            self._save_image(paths[0], "L", 128)
            self._save_image(paths[1], "RGBA", (0, 255, 0, 128))

            scores = self.scorer.score_paths(paths, batch_size=2)

        self.assertEqual(scores["similarities"]["attribute"].shape, (2,))
        torch.testing.assert_close(
            scores["scores"]["attribute"], scores["similarities"]["attribute"]
        )
        self.assertEqual(self.preprocess.modes[-2:], ["RGB", "RGB"])

    def test_invalid_prompts_and_prediction_inputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "positive prompts directly"):
            ZeroShotCLIPConceptScorer(
                "fake",
                "fake",
                {"attribute": {"negative": ["negative"], "positive": ["positive"]}},
                "cpu",
                model=_FakeCLIP(),
                preprocess=self.preprocess,
                tokenizer=self._tokenize,
            )

        with self.assertRaisesRegex(ValueError, "positive prompt"):
            ZeroShotCLIPConceptScorer(
                "fake",
                "fake",
                {"attribute": []},
                "cpu",
                model=_FakeCLIP(),
                preprocess=self.preprocess,
                tokenizer=self._tokenize,
            )

        with self.assertRaisesRegex(ValueError, "No image paths"):
            self.scorer.predict_scores([])
        with self.assertRaisesRegex(ValueError, "Unknown concept"):
            self.scorer.predict_scores([Path("unused.png")], concepts=["missing"])


if __name__ == "__main__":
    unittest.main()
