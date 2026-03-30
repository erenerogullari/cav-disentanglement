import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm
from typing import Tuple
from pathlib import Path

LIBLINEAR_MAX_ITER = 10000
LOGISTIC_MAX_ITER = 10000


def get_latent_encoding_dl(model, dl, layer_name, device, cav_dim=1):
    latent_features = []
    for batch in dl:
        features_batch = (
            get_latent_encoding_batch(model, batch[0].to(device), layer_name)
            .detach()
            .cpu()
        )
        if cav_dim == 1:
            features_batch = (
                features_batch
                if features_batch.dim() == 2
                else features_batch.flatten(start_dim=2).max(2).values
            )
        elif cav_dim == 3:
            features_batch = features_batch.flatten(start_dim=1)
        latent_features.append(features_batch)
    return torch.cat(latent_features)


def get_latent_encoding_batch(model, data, layer_name):
    global layer_act

    # Define Hook
    def get_layer_act_hook_out(m, i, o):
        # returns OUTPUT activations
        global layer_act
        layer_act = o.clone()
        return None

    # Attach hook
    for n, module in model.named_modules():
        if n == layer_name:
            h = module.register_forward_hook(get_layer_act_hook_out)

    # Compute Features
    _ = model(data)
    h.remove()

    return layer_act


def compute_cav(vecs: np.ndarray, targets: np.ndarray, cav_type: str = "svm"):
    """
    Compute a concept activation vector (CAV) for a set of vectors and targets.

    :param vecs:    torch.Tensor of shape (n_samples, n_features)
    :param targets: torch.Tensor of shape (n_samples,)
    :param cav_type:   str, type of CAV to compute. One of ["svm", "ridge", "signal", "mean"]
    :return:       torch.Tensor of shape (1, n_features)
    """

    num_targets = (targets == 1).sum()
    num_notargets = (targets == 0).sum()
    weights = (targets == 1) * 1 / num_targets + (targets == 0) * 1 / num_notargets
    weights = weights / weights.max()

    X = vecs

    if "centered" in cav_type:
        X = X - X.mean(0)[None]

    if "max_scaled" in cav_type:
        max_val = np.abs(X).max(0)
        max_val[max_val == 0] = 1
        X = X / max_val[None]

    if "2mom_scaled" in cav_type:
        scaler = np.mean(X**2) ** 0.5
        X /= scaler

    if "svm" in cav_type:
        linear = LinearSVC(
            random_state=0,
            fit_intercept=True,
            max_iter=LIBLINEAR_MAX_ITER,
        )
        grid_search = GridSearchCV(
            linear, param_grid={"C": [10**i for i in range(-5, 5)]}
        )
        grid_search.fit(X, targets, sample_weight=weights)
        linear = grid_search.best_estimator_
        print("Best C value:", grid_search.best_params_["C"])
        w = torch.Tensor(linear.coef_)

    elif "ridge" in cav_type:
        clf = Ridge(fit_intercept=True)
        grid_search = GridSearchCV(
            clf, param_grid={"alpha": [10**i for i in range(-5, 5)]}
        )
        grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
        clf = grid_search.best_estimator_
        w = torch.tensor(clf.coef_)[None]

    elif "lasso" in cav_type:
        from sklearn.linear_model import Lasso

        clf = Lasso(fit_intercept=True)
        alphas = [10**i for i in range(-5, 1)]
        while True:

            grid_search = GridSearchCV(clf, param_grid={"alpha": alphas})
            grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
            w = torch.tensor(grid_search.best_estimator_.coef_)[None]
            if torch.sqrt((w**2).sum()) != 0:
                break
            else:
                alphas = alphas[:-1]
                if len(alphas) == 0:
                    raise ValueError("Lasso cannot be fit with given alphas.")

    elif "logistic" in cav_type:
        clf = LogisticRegression(
            fit_intercept=True,
            random_state=0,
            max_iter=LOGISTIC_MAX_ITER,
        )
        grid_search = GridSearchCV(clf, param_grid={"C": [10**i for i in range(-5, 5)]})
        grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
        clf = grid_search.best_estimator_
        w = torch.tensor(clf.coef_)

    elif "signal" in cav_type:
        y = targets
        mean_y = y.mean()
        X_residuals = X - X.mean(axis=0)[None]
        covar = (X_residuals * (y - mean_y)[:, np.newaxis]).sum(axis=0) / (
            y.shape[0] - 1
        )
        vary = np.sum((y - mean_y) ** 2, axis=0) / (y.shape[0] - 1)
        w = covar / vary
        w = torch.tensor(w)[None]

    elif "mean" in cav_type:
        w = X[targets == 1].mean(0) - X[targets == 0].mean(0)
        w = torch.tensor(w)[None]

    elif "median" in cav_type:
        w = np.median(X[targets == 1], axis=0) - np.median(X[targets == 0], axis=0)
        w = torch.tensor(w)[None]
    else:
        raise NotImplementedError()

    if "max_scaled" in cav_type:
        w = w * max_val[None]

    if "2mom_scaled" in cav_type:
        w *= scaler

    cav = w / torch.sqrt((w**2).sum())

    return cav


def build_cav_cache_path(
    dataset_name: str,
    model_name: str,
    layer_name: str,
    cav_type: str,
    root_dir: str | Path = "variables",
) -> Path:
    safe_layer = str(layer_name).replace("/", "__").strip() or "no_layer"
    return (
        Path(root_dir)
        / str(dataset_name)
        / str(model_name)
        / safe_layer
        / f"{cav_type}.pth"
    )


def _cache_key(normalize: bool) -> str:
    return "normalized" if normalize else "unnormalized"


def _pack_cache_entry(cavs: torch.Tensor, bias: torch.Tensor) -> dict:
    return {"cavs": cavs.detach().cpu(), "bias": bias.detach().cpu()}


def _unpack_cache_entry(
    entry: dict, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    cavs = entry["cavs"].to(device=device, dtype=dtype)
    bias = entry["bias"].to(device=device, dtype=dtype)
    return cavs, bias


def _normalize_cached(
    cavs: torch.Tensor, bias: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    norms = torch.norm(cavs, dim=1, keepdim=True).clamp_min(1e-12)
    cavs = cavs / norms
    if bias.ndim == 1 and bias.shape[0] == cavs.shape[0]:
        bias = bias / norms.squeeze(1)
    elif bias.ndim == 2 and bias.shape[0] == cavs.shape[0]:
        bias = bias / norms
    return cavs, bias


def _try_load_cached_cavs(
    cache_path: Path,
    normalize: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    if not cache_path.exists():
        return None

    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    key = _cache_key(normalize)

    if isinstance(payload, dict):
        entries = payload.get("entries")
        if isinstance(entries, dict):
            entry = entries.get(key)
            if isinstance(entry, dict) and "cavs" in entry and "bias" in entry:
                return _unpack_cache_entry(entry, device, dtype)

        # Backward compatibility: {"cavs": ..., "bias": ...}
        if "cavs" in payload and "bias" in payload:
            cavs, bias = _unpack_cache_entry(payload, device, dtype)
            if normalize:
                cavs, bias = _normalize_cached(cavs, bias)
            return cavs, bias

    # Backward compatibility: (cavs, bias)
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        cavs = payload[0].to(device=device, dtype=dtype)
        bias = payload[1].to(device=device, dtype=dtype)
        if normalize:
            cavs, bias = _normalize_cached(cavs, bias)
        return cavs, bias

    return None


def _save_cached_cavs(
    cache_path: Path,
    normalize: bool,
    cav_type: str,
    cavs: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key = _cache_key(normalize)

    payload: dict
    if cache_path.exists():
        existing = torch.load(cache_path, map_location="cpu", weights_only=True)
        if isinstance(existing, dict) and isinstance(existing.get("entries"), dict):
            payload = existing
        else:
            payload = {"entries": {}}
    else:
        payload = {"entries": {}}

    payload["type"] = cav_type
    payload["entries"][key] = _pack_cache_entry(cavs, bias)
    torch.save(payload, cache_path)


def compute_cavs(
    vecs: torch.Tensor,
    targets: torch.Tensor,
    type: str = "pattern_cav",
    normalize: bool = True,
    cache_dir: str | Path | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes all concept activation vectors (CAVs) together for a set of vectors and targets.

    :param vecs:    torch.Tensor of shape (n_samples, n_features)
    :param targets: torch.Tensor of shape (n_samples, n_concepts)
    :param type:    str, type of CAV to compute. One of ["pattern_cav", "multi_cav", "svm_cav", "log_cav"]
    :param normalize: bool, whether to normalize the CAVs to unit length
    :param cache_dir: Optional cache path. If provided, CAVs are loaded/saved from/to this file.
    :return:        tuple of (cavs, bias)
    """
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cached = _try_load_cached_cavs(
            cache_path=cache_path,
            normalize=normalize,
            device=vecs.device,
            dtype=vecs.dtype,
        )
        if cached is not None:
            return cached

    mu_x = vecs.mean(dim=0)
    mu_y = targets.mean(dim=0)
    X_centered = vecs - mu_x.unsqueeze(0)  # (n_samples, n_features)
    Y_centered = targets - mu_y.unsqueeze(0)  # (n_samples, n_concepts)
    n_samples = vecs.shape[0]

    if type == "pattern_cav":
        covars = (Y_centered.T @ X_centered) / (
            n_samples - 1
        )  # (n_concepts, n_features)
        vars = torch.sum(Y_centered**2, dim=0) / (n_samples - 1)  # (n_concepts,)
        inv_vars = torch.zeros_like(vars)
        nz = vars > 0
        inv_vars[nz] = 1.0 / vars[nz]
        cavs = covars * inv_vars.unsqueeze(1)  # (n_concepts, n_features)

        if normalize:
            cavs /= torch.norm(cavs, dim=1, keepdim=True)

        # For zero-variance rows, best constant fit is mu_x
        bias = mu_x.unsqueeze(0) - mu_y.unsqueeze(1) * cavs  # (n_concepts, n_features)
        if (~nz).any():
            bias[(~nz), :] = mu_x.unsqueeze(0)

    elif type == "multi_cav":
        y_aug = Y_centered.T @ Y_centered
        cavs = torch.linalg.pinv(y_aug) @ (
            Y_centered.T @ X_centered
        )  # (n_concepts, n_features)
        if normalize:
            cavs /= torch.norm(cavs, dim=1, keepdim=True)
        bias = (mu_x - cavs.T @ mu_y).unsqueeze(0)  # (1, n_features)

    elif type == "svm_cav":
        X = vecs.detach().cpu().numpy()
        Y = targets.detach().cpu().numpy()
        c_grid = [10**i for i in range(-5, 5)]

        cavs_list = []
        bias_list = []
        concept_iterator = tqdm(
            range(Y.shape[1]),
            desc="SVM CAV grid search",
            leave=False,
        )
        for concept_idx in concept_iterator:
            y = Y[:, concept_idx]
            unique_y = np.unique(y)

            if unique_y.shape[0] < 2:
                w = torch.zeros(vecs.shape[1], dtype=vecs.dtype)
                b = torch.tensor(0.0, dtype=vecs.dtype)
            else:
                num_targets = max((y == 1).sum(), 1)
                num_notargets = max((y == 0).sum(), 1)
                sample_weights = (y == 1) * (1.0 / num_targets) + (y == 0) * (
                    1.0 / num_notargets
                )
                sample_weights = sample_weights / sample_weights.max()

                linear = LinearSVC(
                    random_state=0,
                    fit_intercept=True,
                    max_iter=10000,
                )
                grid_search = GridSearchCV(
                    linear,
                    param_grid={"C": c_grid},
                )
                grid_search.fit(X, y, sample_weight=sample_weights)
                linear = grid_search.best_estimator_
                w = torch.tensor(linear.coef_[0], dtype=vecs.dtype)
                b = torch.tensor(float(linear.intercept_[0]), dtype=vecs.dtype)

            cavs_list.append(w)
            bias_list.append(b)

        cavs = torch.stack(cavs_list, dim=0)
        bias = torch.stack(bias_list, dim=0)

        if normalize:
            norms = torch.norm(cavs, dim=1, keepdim=True).clamp_min(1e-12)
            cavs = cavs / norms
            bias = bias / norms.squeeze(1)

        cavs = cavs.to(vecs.device)
        bias = bias.to(vecs.device)

    elif type == "log_cav":
        X = vecs.detach().cpu().numpy()
        Y = targets.detach().cpu().numpy()
        c_grid = [10**i for i in range(-5, 5)]

        cavs_list = []
        bias_list = []
        concept_iterator = tqdm(
            range(Y.shape[1]),
            desc="Logistic CAV grid search",
            leave=False,
        )
        for concept_idx in concept_iterator:
            y = Y[:, concept_idx]
            unique_y = np.unique(y)

            if unique_y.shape[0] < 2:
                w = torch.zeros(vecs.shape[1], dtype=vecs.dtype)
                b = torch.tensor(0.0, dtype=vecs.dtype)
            else:
                num_targets = max((y == 1).sum(), 1)
                num_notargets = max((y == 0).sum(), 1)
                sample_weights = (y == 1) * (1.0 / num_targets) + (y == 0) * (
                    1.0 / num_notargets
                )
                sample_weights = sample_weights / sample_weights.max()

                clf = LogisticRegression(
                    fit_intercept=True,
                    random_state=0,
                    max_iter=LOGISTIC_MAX_ITER,
                )
                grid_search = GridSearchCV(
                    clf,
                    param_grid={"C": c_grid},
                )
                grid_search.fit(X, y, sample_weight=sample_weights)
                clf = grid_search.best_estimator_
                w = torch.tensor(clf.coef_[0], dtype=vecs.dtype)
                b = torch.tensor(float(clf.intercept_[0]), dtype=vecs.dtype)

            cavs_list.append(w)
            bias_list.append(b)

        cavs = torch.stack(cavs_list, dim=0)
        bias = torch.stack(bias_list, dim=0)

        if normalize:
            norms = torch.norm(cavs, dim=1, keepdim=True).clamp_min(1e-12)
            cavs = cavs / norms
            bias = bias / norms.squeeze(1)

        cavs = cavs.to(vecs.device)
        bias = bias.to(vecs.device)

    else:
        raise KeyError(f"Unknown CAV type: {type}")

    if cache_path is not None:
        _save_cached_cavs(
            cache_path=cache_path,
            normalize=normalize,
            cav_type=type,
            cavs=cavs,
            bias=bias,
        )

    return cavs, bias
