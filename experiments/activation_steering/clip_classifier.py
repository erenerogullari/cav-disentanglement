from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from PIL import Image


PromptMap = Mapping[str, Sequence[str]]


class ZeroShotCLIPConceptScorer:
    """Score visual concepts by cosine similarity to positive text prompts."""

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
        for concept, concept_prompts in prompts.items():
            if isinstance(concept_prompts, Mapping):
                raise ValueError(
                    f"Concept '{concept}' must provide positive prompts directly, "
                    "not a positive/negative prompt mapping"
                )
            if isinstance(concept_prompts, str):
                values = [concept_prompts]
            else:
                values = list(concept_prompts)
            if not values or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise ValueError(
                    f"Concept '{concept}' must contain at least one non-empty "
                    "positive prompt"
                )
            validated[concept] = tuple(values)
        return validated

    def _inference_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda")
        return nullcontext()

    def _build_text_prototypes(self):
        prototypes = {}
        with torch.inference_mode(), self._inference_context():
            for concept, concept_prompts in self.prompts.items():
                tokens = self.tokenizer(list(concept_prompts)).to(self.device)
                features = self.model.encode_text(tokens)
                features = torch.nn.functional.normalize(features, dim=-1)
                prototypes[concept] = torch.nn.functional.normalize(
                    features.mean(dim=0), dim=0
                )
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

    def score_paths(
        self,
        image_paths: Sequence[Path],
        batch_size: int = 8,
        concepts: Optional[Sequence[str]] = None,
    ):
        """Return positive-prompt cosine similarities for each concept."""
        image_paths = [Path(path) for path in image_paths]
        if not image_paths:
            raise ValueError("No image paths provided for prediction.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        selected_concepts = self._selected_concepts(concepts)
        similarities = {concept: [] for concept in selected_concepts}

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

                for concept in selected_concepts:
                    concept_similarity = image_features @ self.text_prototypes[concept]
                    similarities[concept].append(
                        concept_similarity.detach().float().cpu()
                    )

        return {
            "scores": {
                concept: torch.cat(values) for concept, values in similarities.items()
            },
            "similarities": {
                concept: torch.cat(values) for concept, values in similarities.items()
            },
        }

    def predict_scores(
        self,
        image_paths: Sequence[Path],
        batch_size: int = 8,
        concepts: Optional[Sequence[str]] = None,
    ):
        """Return positive-prompt cosine similarities in input path order."""
        return self.score_paths(
            image_paths=image_paths,
            batch_size=batch_size,
            concepts=concepts,
        )["scores"]
