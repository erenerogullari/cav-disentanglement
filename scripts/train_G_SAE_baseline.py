from __future__ import annotations

import argparse
import os
import sys
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

MODEL = "vgg16"
LAYER = "features.28"
DATASET = "celeba"

DEFAULT_CONFIG: dict[str, Any] = {
    "name": "G_SAE",
    "wandb_project": "g-sae-training-1",
    "wandb_entity": None,
    "wandb_group": f"g-sae-{DATASET}-{MODEL}-{LAYER}",
    "wandb_mode": os.getenv("WANDB_MODE", "online"),
    "dataset": DATASET,
    "model": MODEL,
    "layer": LAYER,
    "latent_factors": [4.0, 6.0],
    "topk_ratios": [0.15, 0.2, 0.25],
    "direction_source": "decoder",
    "recon_weight": 1.0,
    "cond_weight": 1.0,
    "seed": 23,
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "train_split": 0.8,
    "num_workers": 0,
    "max_samples": None,
    "device": "cuda",
    "check_val_every_n_epoch": 1,
    "save_optional_encoder_directions": False,
}


def import_runtime_dependencies() -> None:
    global Callback
    global DataLoader
    global F
    global G_SAE
    global LearningRateMonitor
    global ModelCheckpoint
    global TensorDataset
    global WandbLogger
    global load_dotenv
    global pl
    global random_split
    global torch
    global wandb

    import pytorch_lightning as pl
    import torch
    import torch.nn.functional as F
    from dotenv import load_dotenv
    from pytorch_lightning.callbacks import (
        Callback,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers import WandbLogger
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import wandb

    sys.path.append(ROOT.as_posix())
    load_dotenv((ROOT / ".env").as_posix())
    DEFAULT_CONFIG["wandb_mode"] = os.getenv("WANDB_MODE", DEFAULT_CONFIG["wandb_mode"])

    from cav_models import G_SAE

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export baseline G_SAE models from a cached latent tensor."
    )
    parser.add_argument(
        "--latent-cache-path",
        type=Path,
        default=None,
        help="Path to a .pth file containing 'encs' and 'labels'.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "notebooks" / "checkpoints" / "g_sae_baseline",
        help="Directory for Lightning checkpoints.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=ROOT / "notebooks" / "checkpoints" / "g_sae_baseline" / "exports",
        help="Directory for exported G_SAE .pth payloads.",
    )
    parser.add_argument(
        "--suggested-baseline-path",
        type=Path,
        default=None,
        help="Metadata-only target path printed in the summary.",
    )
    parser.add_argument("--wandb-project", default=DEFAULT_CONFIG["wandb_project"])
    parser.add_argument("--wandb-entity", default=DEFAULT_CONFIG["wandb_entity"])
    parser.add_argument("--wandb-group", default=DEFAULT_CONFIG["wandb_group"])
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=DEFAULT_CONFIG["wandb_mode"],
    )
    parser.add_argument("--dataset", default=DEFAULT_CONFIG["dataset"])
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"])
    parser.add_argument("--layer", default=DEFAULT_CONFIG["layer"])
    parser.add_argument(
        "--latent-factors",
        type=float,
        nargs="+",
        default=DEFAULT_CONFIG["latent_factors"],
    )
    parser.add_argument(
        "--topk-ratios",
        type=float,
        nargs="+",
        default=DEFAULT_CONFIG["topk_ratios"],
    )
    parser.add_argument(
        "--direction-source",
        choices=["decoder", "encoder"],
        default=DEFAULT_CONFIG["direction_source"],
    )
    parser.add_argument(
        "--recon-weight", type=float, default=DEFAULT_CONFIG["recon_weight"]
    )
    parser.add_argument(
        "--cond-weight", type=float, default=DEFAULT_CONFIG["cond_weight"]
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"]
    )
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"]
    )
    parser.add_argument(
        "--train-split", type=float, default=DEFAULT_CONFIG["train_split"]
    )
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"]
    )
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_CONFIG["max_samples"]
    )
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=DEFAULT_CONFIG["check_val_every_n_epoch"],
    )
    parser.add_argument(
        "--save-optional-encoder-directions",
        action="store_true",
        default=DEFAULT_CONFIG["save_optional_encoder_directions"],
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run the notebook's classifier reconstruction evaluation after training.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["g_sae_val", "dataset_test"],
        default="g_sae_val",
    )
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--classifier-ckpt-path", type=Path, default=None)
    parser.add_argument("--celeba-data-root", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group,
            "wandb_mode": args.wandb_mode,
            "dataset": args.dataset,
            "model": args.model,
            "layer": args.layer,
            "latent_factors": args.latent_factors,
            "topk_ratios": args.topk_ratios,
            "direction_source": args.direction_source,
            "recon_weight": args.recon_weight,
            "cond_weight": args.cond_weight,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "train_split": args.train_split,
            "num_workers": args.num_workers,
            "max_samples": args.max_samples,
            "device": args.device,
            "check_val_every_n_epoch": args.check_val_every_n_epoch,
            "save_optional_encoder_directions": args.save_optional_encoder_directions,
        }
    )
    return config


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parse_args()

import_runtime_dependencies()


def default_latent_cache_path(config: dict[str, Any]) -> Path:
    return (
        ROOT
        / "variables"
        / config["dataset"]
        / config["model"]
        / f"{config['layer']}.pth"
    )


def default_suggested_baseline_path(config: dict[str, Any]) -> Path:
    return (
        ROOT
        / "variables"
        / config["dataset"]
        / config["model"]
        / config["layer"]
        / f"{config['name']}.pth"
    )


def format_decimal(value: float, min_decimals: int = 1, max_decimals: int = 3) -> str:
    formatted = f"{float(value):.{max_decimals}f}".rstrip("0").rstrip(".")
    if "." not in formatted:
        formatted = f"{formatted}." + ("0" * min_decimals)
    else:
        decimals = len(formatted.split(".", 1)[1])
        if decimals < min_decimals:
            formatted = formatted + ("0" * (min_decimals - decimals))
    return formatted


def make_run_name(base_name: str, latent_factor: float, topk_ratio: float) -> str:
    latent_factor_str = format_decimal(latent_factor, min_decimals=1, max_decimals=1)
    topk_ratio_str = format_decimal(topk_ratio, min_decimals=1, max_decimals=3)
    return f"{base_name}-latent_factor={latent_factor_str}-topk_ratio={topk_ratio_str}"


def resolve_trainer_config(device_name: str) -> dict[str, Any]:
    requested = str(device_name).lower()
    if requested in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": 1, "device_label": "cuda"}
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1, "device_label": "mps"}
    return {"accelerator": "cpu", "devices": 1, "device_label": "cpu"}


def normalize_cavs_and_bias(
    cavs: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    norms = torch.norm(cavs, dim=1, keepdim=True).clamp_min(eps)
    cavs_norm = cavs / norms
    bias_norm = bias / norms.squeeze(1)
    return cavs_norm, bias_norm


def get_concept_directions(
    model: G_SAE, source: str = "decoder"
) -> tuple[torch.Tensor, torch.Tensor]:
    if source == "decoder":
        cavs = model.decoder.weight[:, : model.n_concepts].T.detach().cpu().clone()
        bias = torch.zeros(model.n_concepts, dtype=cavs.dtype)
        return cavs, bias

    if source == "encoder":
        cavs = model.encoder.weight[: model.n_concepts, :].detach().cpu().clone()
        if model.encoder.bias is None:
            bias = torch.zeros(model.n_concepts, dtype=cavs.dtype)
        else:
            bias = model.encoder.bias[: model.n_concepts].detach().cpu().clone()
        return cavs, bias

    raise ValueError(f"Unknown direction source: {source}")


def scalarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    scalarized: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                scalarized[key] = float(value.detach().cpu())
            else:
                scalarized[key] = value.detach().cpu().tolist()
        else:
            scalarized[key] = value
    return scalarized


def build_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    loader_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(num_workers > 0),
        "drop_last": False,
    }
    train_generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


class EpochMetricsRecorder(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict[str, float]] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        metric_names = [
            "train_total_loss",
            "train_recon_loss",
            "train_cond_loss",
            "train_latent_density",
            "train_active_latents",
            "val_total_loss",
            "val_recon_loss",
            "val_cond_loss",
            "val_latent_density",
            "val_active_latents",
        ]
        epoch_metrics = {"epoch": int(trainer.current_epoch)}
        for metric_name in metric_names:
            metric_value = trainer.callback_metrics.get(metric_name)
            if metric_value is not None:
                epoch_metrics[metric_name] = float(metric_value.detach().cpu())
        self.history.append(epoch_metrics)


def build_export_payload(
    model: G_SAE,
    config: dict[str, Any],
    run_config: dict[str, Any],
    dataset_metadata: dict[str, int],
    history: list[dict[str, float]],
    best_metrics: dict[str, Any],
) -> dict[str, Any]:
    source = str(run_config["direction_source"])
    cavs_unnorm, bias_unnorm = get_concept_directions(model, source=source)
    cavs_norm, bias_norm = normalize_cavs_and_bias(cavs_unnorm, bias_unnorm)

    metadata = {
        "name": str(config["name"]),
        "run_name": str(run_config["run_name"]),
        "dataset": str(config["dataset"]),
        "model": str(config["model"]),
        "layer": str(config["layer"]),
        "direction_source": source,
        "latent_factor": float(run_config["latent_factor"]),
        "topk_ratio": float(run_config["topk_ratio"]),
        "topk": int(model.topk),
        "n_samples": int(dataset_metadata["n_samples"]),
        "train_size": int(dataset_metadata["train_size"]),
        "val_size": int(dataset_metadata["val_size"]),
        "n_features": int(model.n_features),
        "n_concepts": int(model.n_concepts),
        "n_latents": int(model.n_latents),
        "seed": int(run_config["seed"]),
        "batch_size": int(run_config["batch_size"]),
        "epochs": int(run_config["epochs"]),
        "learning_rate": float(run_config["learning_rate"]),
        "weight_decay": float(run_config["weight_decay"]),
        "recon_weight": float(run_config["recon_weight"]),
        "cond_weight": float(run_config["cond_weight"]),
        "baseline_definition": "guided_sae_alpha_0",
        "latent_cache_path": str(run_config["latent_cache_path"]),
        "export_path": str(run_config["export_path"]),
        "suggested_baseline_path": str(run_config["suggested_baseline_path"]),
        "best_checkpoint_path": str(run_config.get("best_checkpoint_path") or ""),
        "wandb_project": run_config["wandb_project"],
        "wandb_entity": run_config["wandb_entity"],
        "wandb_group": run_config["wandb_group"],
        "wandb_run_id": run_config.get("wandb_run_id"),
        "wandb_run_url": run_config.get("wandb_run_url"),
        "best_metrics": best_metrics,
        "history": history,
    }

    if bool(config["save_optional_encoder_directions"]):
        enc_cavs_unnorm, enc_bias_unnorm = get_concept_directions(
            model, source="encoder"
        )
        enc_cavs_norm, enc_bias_norm = normalize_cavs_and_bias(
            enc_cavs_unnorm, enc_bias_unnorm
        )
        metadata["optional_encoder_export"] = {
            "available": True,
            "direction_source": "encoder",
            "entries": {
                "normalized": {"cavs": enc_cavs_norm, "bias": enc_bias_norm},
                "unnormalized": {"cavs": enc_cavs_unnorm, "bias": enc_bias_unnorm},
            },
        }
    else:
        metadata["optional_encoder_export"] = {"available": False}

    return {
        "type": str(config["name"]),
        "entries": {
            "normalized": {"cavs": cavs_norm, "bias": bias_norm},
            "unnormalized": {"cavs": cavs_unnorm, "bias": bias_unnorm},
        },
        "metadata": metadata,
        "state_dict": model.state_dict(),
    }


class GSAELightningModule(pl.LightningModule):
    def __init__(
        self, config: dict[str, Any], n_concepts: int, n_features: int
    ) -> None:
        super().__init__()
        config = dict(config)
        self.save_hyperparameters(
            {
                "config": config,
                "n_concepts": int(n_concepts),
                "n_features": int(n_features),
            }
        )
        self.config = config
        self.model = G_SAE(
            n_concepts=int(n_concepts),
            n_features=int(n_features),
            device="cpu",
            latent_factor=float(config["latent_factor"]),
            topk_ratio=float(config["topk_ratio"]),
            recon_weight=float(config["recon_weight"]),
            cond_weight=float(config["cond_weight"]),
            direction_source=str(config["direction_source"]),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        _, latents, recons = self.model(x)

        recon_loss = self.model._normalized_recon_mse(recons, x)
        cond_features = latents[:, : self.model.n_concepts]
        cond_loss = F.binary_cross_entropy(cond_features, y)
        total_loss = (
            float(self.config["recon_weight"]) * recon_loss
            + float(self.config["cond_weight"]) * cond_loss
        )

        latent_active_mask = latents > 0
        latent_density = latent_active_mask.float().mean()
        active_latents = latent_active_mask.float().sum(dim=1).mean()

        log_kwargs = {
            "on_step": False,
            "on_epoch": True,
            "logger": True,
            "prog_bar": stage == "val",
            "batch_size": x.shape[0],
        }
        self.log(f"{stage}_total_loss", total_loss, **log_kwargs)
        self.log(f"{stage}_recon_loss", recon_loss, **log_kwargs)
        self.log(f"{stage}_cond_loss", cond_loss, **log_kwargs)
        self.log(f"{stage}_latent_density", latent_density, **log_kwargs)
        self.log(f"{stage}_active_latents", active_latents, **log_kwargs)

        return total_loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )


def load_latent_dataset(config: dict[str, Any], latent_cache_path: Path) -> tuple[
    TensorDataset,
    torch.utils.data.Subset,
    torch.utils.data.Subset,
    dict[str, int],
    int,
    int,
]:
    if not latent_cache_path.exists():
        raise FileNotFoundError(f"Latent cache not found: {latent_cache_path}")

    latent_payload = torch.load(
        latent_cache_path, map_location="cpu", weights_only=True
    )
    encs = latent_payload["encs"].float()
    labels = latent_payload["labels"].float().clamp(min=0)

    if config["max_samples"] is not None:
        max_samples = int(config["max_samples"])
        encs = encs[:max_samples]
        labels = labels[:max_samples]

    n_samples, n_features = encs.shape
    n_concepts = labels.shape[1]

    dataset = TensorDataset(encs, labels)
    train_size = int(len(dataset) * float(config["train_split"]))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Invalid split for {len(dataset)} samples. "
            f"train_size={train_size}, val_size={val_size}."
        )

    split_generator = torch.Generator().manual_seed(int(config["seed"]))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=split_generator
    )

    dataset_metadata = {
        "n_samples": int(n_samples),
        "train_size": int(train_size),
        "val_size": int(val_size),
    }
    return dataset, train_dataset, val_dataset, dataset_metadata, n_features, n_concepts


def configure_wandb(config: dict[str, Any]) -> str | None:
    os.environ["WANDB_MODE"] = str(config["wandb_mode"])
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if str(config["wandb_mode"]) == "online":
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY is missing from .env or the environment.")
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.login(key=wandb_api_key)
    elif wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    return wandb_api_key


def train_sweep(
    config: dict[str, Any],
    latent_cache_path: Path,
    export_dir: Path,
    checkpoint_dir: Path,
    suggested_baseline_path: Path,
) -> dict[str, Any]:
    _, train_dataset, val_dataset, dataset_metadata, n_features, n_concepts = (
        load_latent_dataset(config, latent_cache_path)
    )
    trainer_config = resolve_trainer_config(str(config["device"]))
    run_grid = [
        {"latent_factor": latent_factor, "topk_ratio": topk_ratio}
        for latent_factor, topk_ratio in product(
            config["latent_factors"], config["topk_ratios"]
        )
    ]

    print(f"Latent cache: {latent_cache_path}")
    print(f"Checkpoint root: {checkpoint_dir}")
    print(f"Export dir      : {export_dir}")
    print(f"Suggested move  : {suggested_baseline_path}")
    print(f"W&B project     : {config['wandb_project']}")
    print(f"W&B mode        : {config['wandb_mode']}")
    print("Run grid:")
    for run in run_grid:
        print(f"  latent_factor={run['latent_factor']}, topk_ratio={run['topk_ratio']}")
    print(f"encs shape      : ({dataset_metadata['n_samples']}, {n_features})")
    print(f"labels shape    : ({dataset_metadata['n_samples']}, {n_concepts})")
    print(f"Train size      : {dataset_metadata['train_size']}")
    print(f"Val size        : {dataset_metadata['val_size']}")
    print(
        "Trainer device  :",
        trainer_config["device_label"],
        f"(accelerator={trainer_config['accelerator']}, devices={trainer_config['devices']})",
    )

    configure_wandb(config)

    results = []
    pin_memory = trainer_config["device_label"] == "cuda"

    for sweep_index, sweep_params in enumerate(run_grid, start=1):
        pl.seed_everything(int(config["seed"]), workers=True)

        run_name = make_run_name(
            config["name"],
            latent_factor=float(sweep_params["latent_factor"]),
            topk_ratio=float(sweep_params["topk_ratio"]),
        )
        export_path = export_dir / f"{run_name}.pth"
        run_checkpoint_dir = checkpoint_dir / run_name

        run_config = {
            key: value
            for key, value in config.items()
            if key not in {"latent_factors", "topk_ratios"}
        }
        run_config.update(sweep_params)
        run_config.update(
            {
                "run_name": run_name,
                "export_path": export_path.as_posix(),
                "suggested_baseline_path": suggested_baseline_path.as_posix(),
                "checkpoint_dir": run_checkpoint_dir.as_posix(),
                "latent_cache_path": latent_cache_path.as_posix(),
            }
        )

        train_loader, val_loader = build_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=int(config["batch_size"]),
            num_workers=int(config["num_workers"]),
            seed=int(config["seed"]),
            pin_memory=pin_memory,
        )

        history_callback = EpochMetricsRecorder()
        checkpoint_callback = ModelCheckpoint(
            dirpath=run_checkpoint_dir,
            filename=run_name + "-{epoch:02d}-{val_total_loss:.4f}",
            monitor="val_total_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
            verbose=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        wandb_logger = WandbLogger(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            name=run_name,
            group=config["wandb_group"],
            job_type="train",
            save_dir=ROOT.as_posix(),
            log_model=False,
            reinit=True,
            tags=["G_SAE", config["dataset"], config["model"], config["layer"]],
        )

        lightning_model = GSAELightningModule(
            config=run_config,
            n_concepts=n_concepts,
            n_features=n_features,
        )
        run_hparams = {
            **run_config,
            **dataset_metadata,
            "n_features": int(n_features),
            "n_concepts": int(n_concepts),
            "n_latents": int(lightning_model.model.n_latents),
            "topk": int(lightning_model.model.topk),
            "accelerator": trainer_config["accelerator"],
            "devices": trainer_config["devices"],
            "grid_size": len(run_grid),
        }
        wandb_logger.log_hyperparams(run_hparams)

        trainer = pl.Trainer(
            max_epochs=int(config["epochs"]),
            accelerator=trainer_config["accelerator"],
            devices=trainer_config["devices"],
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor, history_callback],
            deterministic=True,
            log_every_n_steps=1,
            enable_progress_bar=True,
            enable_model_summary=True,
            check_val_every_n_epoch=int(config["check_val_every_n_epoch"]),
            default_root_dir=ROOT.as_posix(),
        )

        try:
            print(f"\n[{sweep_index}/{len(run_grid)}] Training {run_name}")
            trainer.fit(
                lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

            best_model_path = checkpoint_callback.best_model_path
            if best_model_path:
                best_module = GSAELightningModule.load_from_checkpoint(
                    best_model_path,
                    config=run_config,
                    n_concepts=n_concepts,
                    n_features=n_features,
                )
            else:
                best_module = lightning_model

            best_val_metrics_raw = trainer.validate(
                best_module, dataloaders=val_loader, verbose=False
            )[0]
            best_val_metrics = scalarize_metrics(best_val_metrics_raw)
            best_score = checkpoint_callback.best_model_score
            best_val_total_loss = (
                float(best_score.detach().cpu())
                if best_score is not None
                else float(best_val_metrics["val_total_loss"])
            )

            run_config["best_checkpoint_path"] = best_model_path
            run_config["wandb_run_id"] = wandb_logger.experiment.id
            run_config["wandb_run_url"] = wandb_logger.experiment.url

            payload = build_export_payload(
                model=best_module.model.to("cpu"),
                config=config,
                run_config=run_config,
                dataset_metadata=dataset_metadata,
                history=history_callback.history,
                best_metrics=best_val_metrics,
            )
            export_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, export_path)

            wandb_logger.experiment.summary["export_path"] = export_path.as_posix()
            wandb_logger.experiment.summary["best_checkpoint_path"] = best_model_path
            wandb_logger.experiment.summary["n_latents"] = int(
                best_module.model.n_latents
            )
            wandb_logger.experiment.summary["topk"] = int(best_module.model.topk)
            for metric_name, metric_value in best_val_metrics.items():
                wandb_logger.experiment.summary[f"best_{metric_name}"] = metric_value

            result = {
                "run_name": run_name,
                "latent_factor": float(sweep_params["latent_factor"]),
                "topk_ratio": float(sweep_params["topk_ratio"]),
                "n_latents": int(best_module.model.n_latents),
                "topk": int(best_module.model.topk),
                "best_val_total_loss": float(best_val_total_loss),
                "best_val_recon_loss": float(
                    best_val_metrics.get("val_recon_loss", float("nan"))
                ),
                "best_val_cond_loss": float(
                    best_val_metrics.get("val_cond_loss", float("nan"))
                ),
                "best_val_latent_density": float(
                    best_val_metrics.get("val_latent_density", float("nan"))
                ),
                "best_val_active_latents": float(
                    best_val_metrics.get("val_active_latents", float("nan"))
                ),
                "export_path": export_path.as_posix(),
                "checkpoint_path": best_model_path,
                "wandb_run_url": wandb_logger.experiment.url,
            }
            results.append(result)
            print(
                f"[{sweep_index}/{len(run_grid)}] Saved {run_name} to {export_path} | "
                f"best val_total_loss={result['best_val_total_loss']:.6f}"
            )
        finally:
            wandb.finish()

    if not results:
        raise RuntimeError("No sweep runs were recorded.")

    results = sorted(results, key=lambda item: item["best_val_total_loss"])
    best_run = results[0]
    best_export_path = Path(best_run["export_path"])

    print("\nSweep summary (sorted by val_total_loss):")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:02d}. {result['run_name']} | "
            f"val_total={result['best_val_total_loss']:.6f} | "
            f"val_recon={result['best_val_recon_loss']:.6f} | "
            f"val_cond={result['best_val_cond_loss']:.6f} | "
            f"n_latents={result['n_latents']} | topk={result['topk']}"
        )

    print()
    print(f"Best run              : {best_run['run_name']}")
    print(f"Best run export       : {best_export_path}")
    print(f"Suggested move target : {suggested_baseline_path}")
    print(f"Best W&B run URL      : {best_run['wandb_run_url']}")

    validate_exports(results, n_concepts=n_concepts, n_features=n_features)

    return {
        "results": results,
        "best_run": best_run,
        "best_export_path": best_export_path,
        "val_dataset": val_dataset,
        "n_features": n_features,
        "n_concepts": n_concepts,
    }


def validate_exports(
    results: list[dict[str, Any]], n_concepts: int, n_features: int
) -> None:
    for result in results:
        export_path = Path(result["export_path"])
        if not export_path.exists():
            raise FileNotFoundError(f"Missing export file: {export_path}")
        reloaded = torch.load(export_path, map_location="cpu", weights_only=True)

        if reloaded["type"] != "G_SAE":
            raise ValueError(f"Unexpected type in {export_path}: {reloaded['type']}")
        cavs = reloaded["entries"]["normalized"]["cavs"]
        bias = reloaded["entries"]["normalized"]["bias"]
        if cavs.shape != (n_concepts, n_features):
            raise ValueError(
                f"Expected cavs shape {(n_concepts, n_features)}, "
                f"got {tuple(cavs.shape)} for {export_path}"
            )
        if bias.shape[0] != n_concepts:
            raise ValueError(
                f"Expected bias length {n_concepts}, got {bias.shape[0]} for {export_path}"
            )
        norms = torch.norm(cavs, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
            raise ValueError(f"Normalized cavs are not unit norm for {export_path}")

    print("Validated all checkpoint exports.")
    print(f"Validated {len(results)} parameter-specific exports.")


def run_evaluation(
    args: argparse.Namespace,
    config: dict[str, Any],
    best_export_path: Path,
    suggested_baseline_path: Path,
    val_dataset,
    n_features: int,
    n_concepts: int,
) -> None:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from torch.utils.data import Subset

    from datasets import get_dataset
    from models import get_fn_model_loader

    eval_config = {
        "split": args.eval_split,
        "batch_size": args.eval_batch_size,
        "num_workers": int(config["num_workers"]),
        "device": config["device"],
        "threshold": 0.5,
        "positive_class": 1,
        "activation_spatial_mode": "per_location",
        "g_sae_export_path": best_export_path,
        "classifier_ckpt_candidates": [
            args.classifier_ckpt_path,
            ROOT
            / "checkpoints"
            / f"checkpoint_{config['model']}_{config['dataset']}.pth",
            Path("/media/erogullari/checkpoints")
            / f"checkpoint_{config['model']}_{config['dataset']}.pth",
        ],
        "data_path_candidates": [
            args.celeba_data_root,
            os.getenv("CELEBA_DATA_ROOT"),
            "/Users/erogullari/datasets/",
            "/home/erogullari/datasets/",
        ],
        "dataset_kwargs": {
            "normalize_data": True,
            "image_size": 224,
            "attacked_classes": [1],
            "p_artifact": 0.4,
            "artifact_type": "ch_time",
            "time_format": "datetime",
            "entanglement_factor": 5,
        },
    }

    def resolve_eval_device(device_name: str) -> torch.device:
        requested = str(device_name).lower()
        if requested in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
            return torch.device("cuda")
        if requested in {"auto", "mps"} and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def first_existing_path(candidates, *, kind: str) -> Path:
        checked = []
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate).expanduser()
            checked.append(path)
            if path.exists():
                return path
        checked_text = "\n".join(f"  - {p}" for p in checked)
        raise FileNotFoundError(f"Could not find {kind}. Checked:\n{checked_text}")

    def load_classifier_for_eval(
        model_name: str,
        ckpt_path: Path,
        num_classes: int,
        device: torch.device,
    ) -> torch.nn.Module:
        model_loader = get_fn_model_loader(model_name)
        model = model_loader(n_class=int(num_classes), pretrained=False)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            key = key.removeprefix("module.").removeprefix("model.")
            key = key.replace("classifier.last", "classifier.6")
            cleaned_state_dict[key] = value

        model.load_state_dict(cleaned_state_dict)
        model.to(device)
        model.eval()
        return model

    def load_g_sae_for_eval(export_path: Path, device: torch.device) -> G_SAE:
        payload = torch.load(export_path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        n_features_export = int(metadata.get("n_features", n_features))
        n_concepts_export = int(metadata.get("n_concepts", n_concepts))

        model = G_SAE(
            n_concepts=n_concepts_export,
            n_features=n_features_export,
            device="cpu",
            latent_factor=float(
                metadata.get("latent_factor", config["latent_factors"][0])
            ),
            topk_ratio=float(metadata.get("topk_ratio", config["topk_ratios"][0])),
            n_latents=int(metadata["n_latents"]) if "n_latents" in metadata else None,
            topk=int(metadata["topk"]) if "topk" in metadata else None,
            recon_weight=float(metadata.get("recon_weight", config["recon_weight"])),
            cond_weight=float(metadata.get("cond_weight", config["cond_weight"])),
            direction_source=str(
                metadata.get("direction_source", config["direction_source"])
            ),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        return model

    def get_eval_dataset(celeba_data_root: Path) -> Subset:
        dataset_fn = get_dataset(config["dataset"])
        full_dataset = dataset_fn(
            data_paths=[celeba_data_root.as_posix()],
            **eval_config["dataset_kwargs"],
        )

        split = str(eval_config["split"])
        if split == "g_sae_val":
            indices = list(val_dataset.indices)
        elif split == "dataset_test":
            indices = list(full_dataset.idxs_test)
        else:
            raise ValueError(f"Unknown evaluation split: {split}")

        return Subset(full_dataset, indices)

    def get_named_module(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
        modules = dict(model.named_modules())
        if layer_name not in modules:
            available = "\n".join(modules.keys())
            raise KeyError(
                f"Layer '{layer_name}' not found. Available modules:\n{available}"
            )
        return modules[layer_name]

    def reconstruct_activation_with_g_sae(
        activation: torch.Tensor,
        g_sae: G_SAE,
        spatial_mode: str = "per_location",
        chunk_size: int = 8192,
    ) -> torch.Tensor:
        original_shape = activation.shape
        original_dtype = activation.dtype
        model_dtype = next(g_sae.parameters()).dtype
        n_features_sae = int(g_sae.n_features)

        if activation.ndim == 2 and activation.shape[-1] == n_features_sae:
            flat = activation
            restore = lambda x: x.reshape(original_shape)
        elif (
            activation.ndim > 2 and int(np.prod(activation.shape[1:])) == n_features_sae
        ):
            flat = activation.reshape(activation.shape[0], n_features_sae)
            restore = lambda x: x.reshape(original_shape)
        elif activation.ndim > 2 and activation.shape[-1] == n_features_sae:
            flat = activation.reshape(-1, n_features_sae)
            restore = lambda x: x.reshape(original_shape)
        elif activation.ndim > 2 and activation.shape[1] == n_features_sae:
            if spatial_mode != "per_location":
                raise ValueError(
                    f"Unsupported spatial_mode for channel-first activation: {spatial_mode}"
                )
            moved = activation.movedim(1, -1).contiguous()
            moved_shape = moved.shape

            def restore(x):
                return x.reshape(moved_shape).movedim(-1, 1).reshape(original_shape)

            flat = moved.reshape(-1, n_features_sae)
        else:
            raise ValueError(
                f"Cannot apply G_SAE with n_features={n_features_sae} "
                f"to activation shape {tuple(original_shape)}"
            )

        recons = []
        for start in range(0, flat.shape[0], int(chunk_size)):
            flat_chunk = flat[start : start + int(chunk_size)].to(dtype=model_dtype)
            _, _, recon_chunk = g_sae(flat_chunk)
            recons.append(recon_chunk.to(dtype=original_dtype))
        return restore(torch.cat(recons, dim=0))

    def classifier_probabilities(
        logits: torch.Tensor, positive_class: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim == 1 or logits.shape[-1] == 1:
            probs_pos = torch.sigmoid(logits.reshape(-1))
            preds = (probs_pos >= float(eval_config["threshold"])).long()
            return probs_pos, preds

        probs = torch.softmax(logits, dim=1)
        probs_pos = probs[:, int(positive_class)]
        preds = probs.argmax(dim=1)
        return probs_pos, preds

    def compute_binary_metrics(y_true, y_prob, y_pred) -> dict[str, float | int]:
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        y_pred = np.asarray(y_pred).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        metrics = {
            "n_samples": int(y_true.shape[0]),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        try:
            metrics["auroc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auroc"] = float("nan")

        try:
            metrics["average_precision"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["average_precision"] = float("nan")

        return metrics

    def evaluate_classifier(
        classifier: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        hook_layer: str | None = None,
        g_sae: G_SAE | None = None,
    ) -> dict[str, float | int]:
        y_true_all = []
        y_prob_all = []
        y_pred_all = []
        hook_handle = None

        if hook_layer is not None:
            if g_sae is None:
                raise ValueError("g_sae must be provided when hook_layer is set.")

            layer = get_named_module(classifier, hook_layer)

            def hook_fn(module, inputs, output):
                if not torch.is_tensor(output):
                    raise TypeError(
                        f"Expected tensor output from {hook_layer}, got {type(output)}"
                    )
                return reconstruct_activation_with_g_sae(
                    output,
                    g_sae,
                    spatial_mode=str(eval_config["activation_spatial_mode"]),
                )

            hook_handle = layer.register_forward_hook(hook_fn)

        try:
            with torch.inference_mode():
                for x, y in dataloader:
                    x = x.to(device)
                    y = y.long().view(-1)
                    logits = classifier(x)
                    probs_pos, preds = classifier_probabilities(
                        logits,
                        positive_class=int(eval_config["positive_class"]),
                    )
                    y_true_all.append(y.cpu())
                    y_prob_all.append(probs_pos.detach().cpu())
                    y_pred_all.append(preds.detach().cpu())
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        y_true = torch.cat(y_true_all).numpy()
        y_prob = torch.cat(y_prob_all).numpy()
        y_pred = torch.cat(y_pred_all).numpy()
        return compute_binary_metrics(y_true, y_prob, y_pred)

    eval_device = resolve_eval_device(str(eval_config["device"]))
    classifier_ckpt_path = first_existing_path(
        eval_config["classifier_ckpt_candidates"],
        kind="classifier checkpoint",
    )
    g_sae_eval_path = first_existing_path(
        [eval_config["g_sae_export_path"], suggested_baseline_path],
        kind="G_SAE export",
    )
    celeba_data_root = first_existing_path(
        eval_config["data_path_candidates"], kind="CelebA data root"
    )

    print(f"Evaluation device      : {eval_device}")
    print(f"Classifier checkpoint : {classifier_ckpt_path}")
    print(f"G_SAE export          : {g_sae_eval_path}")
    print(f"CelebA data root      : {celeba_data_root}")

    eval_dataset = get_eval_dataset(celeba_data_root)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(eval_config["batch_size"]),
        shuffle=False,
        num_workers=int(eval_config["num_workers"]),
        pin_memory=eval_device.type == "cuda",
    )

    classifier = load_classifier_for_eval(
        model_name=config["model"],
        ckpt_path=classifier_ckpt_path,
        num_classes=2,
        device=eval_device,
    )
    g_sae_eval = load_g_sae_for_eval(g_sae_eval_path, eval_device)

    baseline_metrics = evaluate_classifier(
        classifier=classifier,
        dataloader=eval_loader,
        device=eval_device,
    )
    g_sae_metrics = evaluate_classifier(
        classifier=classifier,
        dataloader=eval_loader,
        device=eval_device,
        hook_layer=config["layer"],
        g_sae=g_sae_eval,
    )

    evaluation_df = pd.DataFrame(
        [baseline_metrics, g_sae_metrics],
        index=["classifier_baseline", "classifier_with_G_SAE_reconstruction"],
    )
    delta_row = (
        evaluation_df.loc["classifier_with_G_SAE_reconstruction"]
        - evaluation_df.loc["classifier_baseline"]
    )
    evaluation_df.loc["delta_G_SAE_minus_baseline"] = delta_row

    print(evaluation_df.round(6).to_string())
    print(
        "Accuracy drop:",
        f"{evaluation_df.loc['classifier_baseline', 'accuracy']} - "
        f"{evaluation_df.loc['classifier_with_G_SAE_reconstruction', 'accuracy']:.6f} = ",
        f"{evaluation_df.loc['delta_G_SAE_minus_baseline', 'accuracy']:.6f}",
    )
    print(
        "F1 drop:",
        f"{evaluation_df.loc['classifier_baseline', 'f1']} - "
        f"{evaluation_df.loc['classifier_with_G_SAE_reconstruction', 'f1']:.6f} = ",
        f"{evaluation_df.loc['delta_G_SAE_minus_baseline', 'f1']:.6f}",
    )
    print(
        "AUROC drop:",
        f"{evaluation_df.loc['classifier_baseline', 'auroc']} - "
        f"{evaluation_df.loc['classifier_with_G_SAE_reconstruction', 'auroc']:.6f} = ",
        f"{evaluation_df.loc['delta_G_SAE_minus_baseline', 'auroc']:.6f}",
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    latent_cache_path = args.latent_cache_path or default_latent_cache_path(config)
    suggested_baseline_path = (
        args.suggested_baseline_path or default_suggested_baseline_path(config)
    )

    training_outputs = train_sweep(
        config=config,
        latent_cache_path=latent_cache_path.expanduser().resolve(),
        export_dir=args.export_dir.expanduser().resolve(),
        checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
        suggested_baseline_path=suggested_baseline_path.expanduser().resolve(),
    )

    if args.run_eval:
        run_evaluation(
            args=args,
            config=config,
            best_export_path=training_outputs["best_export_path"],
            suggested_baseline_path=suggested_baseline_path.expanduser().resolve(),
            val_dataset=training_outputs["val_dataset"],
            n_features=training_outputs["n_features"],
            n_concepts=training_outputs["n_concepts"],
        )


if __name__ == "__main__":
    main()
