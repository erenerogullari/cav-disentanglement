import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf

from datasets.celeba.celeba_attacked import get_active_artifact_concepts
from experiments.utils.train_model import train_classifier


def build_experiment_signature(cfg: DictConfig) -> dict[str, Any]:
    active_concepts = get_active_artifact_concepts(int(cfg.dataset.num_concepts))
    return {
        "schema_version": 1,
        "dataset": {
            "name": str(cfg.dataset.name),
            "num_concepts": int(cfg.dataset.num_concepts),
            "active_concepts": active_concepts,
            "attacked_classes": [
                int(value) for value in cfg.dataset.attacked_classes
            ],
            "timestamp_probability_attacked": float(cfg.dataset.p_artifact),
            "timestamp_probability_clean": 0.005,
            "cooccurrence_probability": float(
                cfg.dataset.cooccurrence_probability
            ),
            "entanglement_factor": float(cfg.dataset.entanglement_factor),
            "artifact_seed": int(cfg.dataset.artifact_seed),
            "split_seed": int(cfg.dataset.seed),
            "val_split": float(cfg.dataset.val_split),
            "test_split": float(cfg.dataset.test_split),
            "image_size": int(cfg.dataset.image_size),
            "artifact_type": str(cfg.dataset.artifact_type),
            "time_format": str(cfg.dataset.time_format),
            "brightness_factor": float(cfg.dataset.brightness_factor),
            "normalize_data": bool(cfg.dataset.normalize_data),
        },
        "classifier": {
            "model": str(cfg.model.name),
            "clean_checkpoint": str(
                Path(cfg.model.clean_ckpt_path).expanduser().resolve()
            ),
            "random_seed": int(cfg.classifier_train.random_seed),
            "num_epochs": int(cfg.classifier_train.num_epochs),
            "learning_rate": float(cfg.classifier_train.learning_rate),
            "batch_size": int(cfg.classifier_train.batch_size),
            "val_split": float(cfg.classifier_train.val_split),
            "test_split": float(cfg.classifier_train.test_split),
            "save_best": bool(cfg.classifier_train.save_best),
            "max_samples_per_split": cfg.classifier_train.get(
                "max_samples_per_split", None
            ),
        },
    }


def canonical_signature_json(signature: dict[str, Any]) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def build_variant_id(signature: dict[str, Any]) -> str:
    dataset_signature = signature["dataset"]
    concept_slug = "-".join(dataset_signature["active_concepts"])
    digest = hashlib.sha256(
        canonical_signature_json(signature).encode("utf-8")
    ).hexdigest()[:10]
    return f"k{dataset_signature['num_concepts']}-{concept_slug}-{digest}"


def classifier_checkpoint_path(repository_root: Path, variant_id: str) -> Path:
    return (
        repository_root
        / "checkpoints"
        / "celeba_attacked"
        / variant_id
        / "checkpoint_vgg16.pth"
    )


def validate_classifier_checkpoint(
    checkpoint_path: Path, expected_signature: dict[str, Any]
) -> None:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError(
            f"Could not load classifier checkpoint '{checkpoint_path}'."
        ) from exc

    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        raise ValueError(
            f"Classifier checkpoint '{checkpoint_path}' has no model_state_dict."
        )
    actual_signature = checkpoint.get("experiment_signature")
    if actual_signature != expected_signature:
        raise ValueError(
            "Classifier checkpoint signature mismatch at "
            f"'{checkpoint_path}'. Refusing to overwrite the cached model."
        )


def _build_classifier_config(cfg: DictConfig) -> DictConfig:
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(dataset_cfg, dict)
    dataset_cfg.pop("_target_", None)
    dataset_cfg.pop("cache_namespace", None)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)
    model_cfg["ckpt_path"] = model_cfg.pop("clean_ckpt_path")

    train_cfg = OmegaConf.to_container(cfg.classifier_train, resolve=True)
    assert isinstance(train_cfg, dict)
    return OmegaConf.create(
        {
            "experiment": {"name": "multi_concept_alignment_classifier"},
            "dataset": dataset_cfg,
            "model": model_cfg,
            "train": train_cfg,
        }
    )


def ensure_classifier_checkpoint(
    cfg: DictConfig,
    repository_root: Path,
    signature: dict[str, Any],
    variant_id: str,
    train_fn: Callable[..., Path] = train_classifier,
) -> tuple[Path, bool]:
    checkpoint_path = classifier_checkpoint_path(repository_root, variant_id)
    if checkpoint_path.exists():
        validate_classifier_checkpoint(checkpoint_path, signature)
        return checkpoint_path, False

    classifier_cfg = _build_classifier_config(cfg)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_fn(
        classifier_cfg,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata={"experiment_signature": signature},
        media_dir=checkpoint_path.parent / "media",
    )
    validate_classifier_checkpoint(checkpoint_path, signature)
    return checkpoint_path, True
