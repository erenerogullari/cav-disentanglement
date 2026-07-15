from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from PIL import Image


PromptMap = Mapping[str, Mapping[str, Sequence[str]]]


class ZeroShotCLIPConceptScorer:
    """Score binary visual concepts with paired CLIP text prompts."""

    _CLASS_NAMES = ("negative", "positive")

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        prompts: PromptMap,
        device: Optional[str] = None,
        *,
        model=None,
        preprocess=None,
        tokenizer=None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.prompts = self._validate_prompts(prompts)

        supplied = (model is not None, preprocess is not None, tokenizer is not None)
        if any(supplied) and not all(supplied):
            raise ValueError(
                "model, preprocess, and tokenizer must be supplied together"
            )

        if model is None:
            try:
                import open_clip
            except ImportError as error:
                raise ImportError(
                    "open_clip is required; install open_clip_torch==3.3.0"
                ) from error

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device,
                force_quick_gelu=pretrained == "openai",
            )
            tokenizer = open_clip.get_tokenizer(model_name)

        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.text_prototypes = self._build_text_prototypes()

    @classmethod
    def _validate_prompts(cls, prompts: PromptMap):
        if not prompts:
            raise ValueError("prompts must contain at least one concept")

        validated = {}
        for concept, class_prompts in prompts.items():
            missing = [name for name in cls._CLASS_NAMES if name not in class_prompts]
            if missing:
                raise ValueError(
                    f"Concept '{concept}' is missing prompt classes: {missing}"
                )

            validated[concept] = {}
            for class_name in cls._CLASS_NAMES:
                values = list(class_prompts[class_name])
                if not values or not all(
                    isinstance(value, str) and value for value in values
                ):
                    raise ValueError(
                        f"Concept '{concept}' class '{class_name}' must contain "
                        "at least one non-empty prompt"
                    )
                validated[concept][class_name] = tuple(values)
        return validated

    def _inference_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda")
        return nullcontext()

    def _build_text_prototypes(self):
        prototypes = {}
        with torch.inference_mode(), self._inference_context():
            for concept, class_prompts in self.prompts.items():
                class_prototypes = []
                for class_name in self._CLASS_NAMES:
                    tokens = self.tokenizer(list(class_prompts[class_name])).to(
                        self.device
                    )
                    features = self.model.encode_text(tokens)
                    features = torch.nn.functional.normalize(features, dim=-1)
                    prototype = torch.nn.functional.normalize(
                        features.mean(dim=0), dim=0
                    )
                    class_prototypes.append(prototype)
                prototypes[concept] = torch.stack(class_prototypes)
        return prototypes

    def _selected_concepts(self, concepts):
        if concepts is None:
            return tuple(self.prompts)
        concepts = tuple(concepts)
        unknown = sorted(set(concepts) - set(self.prompts))
        if unknown:
            raise ValueError(
                f"Unknown concept(s): {unknown}. Available concepts: "
                f"{sorted(self.prompts)}"
            )
        return concepts

    def _logit_scale(self):
        logit_scale = getattr(self.model, "logit_scale", None)
        if logit_scale is None:
            return torch.tensor(100.0, device=self.device)
        return logit_scale.exp()

    def score_paths(
        self,
        image_paths: Sequence[Path],
        batch_size: int = 8,
        concepts: Optional[Sequence[str]] = None,
    ):
        """Return positive probabilities and cosine margins for each concept."""
        image_paths = [Path(path) for path in image_paths]
        if not image_paths:
            raise ValueError("No image paths provided for prediction.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        selected_concepts = self._selected_concepts(concepts)
        probabilities = {concept: [] for concept in selected_concepts}
        margins = {concept: [] for concept in selected_concepts}

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(self.preprocess(image.convert("RGB")))
            batch = torch.stack(images).to(self.device)

            with torch.inference_mode(), self._inference_context():
                image_features = self.model.encode_image(batch)
                image_features = torch.nn.functional.normalize(
                    image_features, dim=-1
                )
                logit_scale = self._logit_scale()

                for concept in selected_concepts:
                    similarities = image_features @ self.text_prototypes[concept].T
                    concept_probabilities = (logit_scale * similarities).softmax(
                        dim=-1
                    )[:, 1]
                    concept_margins = similarities[:, 1] - similarities[:, 0]
                    probabilities[concept].append(
                        concept_probabilities.detach().float().cpu()
                    )
                    margins[concept].append(concept_margins.detach().float().cpu())

        return {
            "probabilities": {
                concept: torch.cat(values)
                for concept, values in probabilities.items()
            },
            "margins": {
                concept: torch.cat(values) for concept, values in margins.items()
            },
        }

    def predict_proba(
        self,
        image_paths: Sequence[Path],
        batch_size: int = 8,
        concepts: Optional[Sequence[str]] = None,
    ):
        """Return positive-class probabilities in the input path order."""
        return self.score_paths(
            image_paths=image_paths,
            batch_size=batch_size,
            concepts=concepts,
        )["probabilities"]
