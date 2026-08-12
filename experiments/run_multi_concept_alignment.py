import json
import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

from experiments.concept_alignment import evaluate_concept_alignment
from experiments.model_correction.dir_model import get_dir_models_with_data
from experiments.multi_concept_alignment.localization import (
    evaluate_multi_concept_localization,
)
from experiments.multi_concept_alignment.variant import (
    build_experiment_signature,
    build_variant_id,
    ensure_classifier_checkpoint,
)
from experiments.utils.utils import get_save_dir
from utils.cav import build_cav_cache_path

log = logging.getLogger(__name__)


def _configure_variant(cfg: DictConfig, repository_root: Path):
    if cfg.model.name != "vgg16":
        raise ValueError("The multi-concept proof of concept supports only VGG16.")
    if list(cfg.cav.target_concepts):
        raise ValueError("This experiment requires cav.target_concepts to be empty.")
    if cfg.cav.get("beta", None) is not None:
        raise ValueError("This experiment requires cav.beta=null.")
    if bool(cfg.localization.concept_cleaning.enabled):
        raise ValueError("Cleaned-CAV variants are disabled for this proof of concept.")

    signature = build_experiment_signature(cfg)
    variant_id = build_variant_id(signature)
    checkpoint_path, trained = ensure_classifier_checkpoint(
        cfg, repository_root, signature, variant_id
    )
    with open_dict(cfg):
        cfg.experiment.variant_id = variant_id
        cfg.dataset.cache_namespace = f"celeba_attacked/{variant_id}"
        cfg.model.ckpt_path = str(checkpoint_path)
        cfg.experiment.out = str(
            repository_root
            / "results"
            / cfg.experiment.name
            / variant_id
            / cfg.model.name
            / cfg.cav.layer
            / cfg.cav.name
        )
    return signature, variant_id, checkpoint_path, trained


def _write_run_manifest(
    cfg: DictConfig,
    repository_root: Path,
    signature: dict,
    checkpoint_path: Path,
    classifier_trained: bool,
) -> Path:
    run_dir = get_save_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    activation_cache = (
        repository_root
        / "variables"
        / cfg.dataset.cache_namespace
        / cfg.model.name
        / f"{cfg.cav.layer}.pth"
    )
    baseline_cav_cache = repository_root / build_cav_cache_path(
        dataset_name=cfg.dataset.cache_namespace,
        model_name=cfg.model.name,
        layer_name=cfg.cav.layer,
        cav_type=cfg.cav.name,
        random_seed=cfg.train.random_seed,
    )
    manifest = {
        "experiment_signature": signature,
        "variant_id": str(cfg.experiment.variant_id),
        "cav": {
            "method": str(cfg.cav.name),
            "alpha": float(cfg.cav.alpha),
            "layer": str(cfg.cav.layer),
            "mode": str(cfg.cav.cav_mode),
            "training_epochs": int(cfg.train.num_epochs),
            "learning_rate": float(cfg.train.learning_rate),
            "optimal_init": bool(cfg.cav.optimal_init),
            "target_concepts": list(cfg.cav.target_concepts),
            "beta": cfg.cav.get("beta", None),
            "max_preprocessing_samples": cfg.train.get(
                "max_preprocessing_samples", None
            ),
        },
        "cache_paths": {
            "classifier": str(checkpoint_path),
            "activation": str(activation_cache),
            "baseline_cav": str(baseline_cav_cache),
            "direction_model": str(run_dir / "state_dict.pth"),
        },
        "classifier_trained_in_this_invocation": classifier_trained,
        "output_paths": {
            "alignment": str(run_dir / "alignment"),
            "localization": str(run_dir / "localization"),
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    return manifest_path


@hydra.main(
    version_base=None,
    config_path="../experiment_configs",
    config_name="multi_concept_alignment",
)
def run(cfg: DictConfig) -> None:
    repository_root = Path(get_original_cwd())
    signature, variant_id, checkpoint_path, trained = _configure_variant(
        cfg, repository_root
    )
    log.info(
        "Running variant %s with classifier cache %s (%s).",
        variant_id,
        checkpoint_path,
        "trained" if trained else "reused",
    )

    direction_model, baseline_model, _, _ = get_dir_models_with_data(cfg)
    run_dir = get_save_dir(cfg)
    evaluate_concept_alignment(
        cfg,
        direction_model,
        baseline_model,
        output_dir=run_dir / "alignment",
    )
    evaluate_multi_concept_localization(
        cfg,
        direction_model,
        baseline_model,
        output_dir=run_dir / "localization",
    )
    manifest_path = _write_run_manifest(
        cfg,
        repository_root,
        signature,
        checkpoint_path,
        trained,
    )
    log.info("Experiment completed. Manifest: %s", manifest_path)


if __name__ == "__main__":
    run()
