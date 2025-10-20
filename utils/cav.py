import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import tqdm
from typing import Tuple

def get_latent_encoding_dl(model, dl, layer_name, device, cav_dim=1):
    latent_features = []
    for batch in dl:
        features_batch = get_latent_encoding_batch(model, batch[0].to(device), layer_name).detach().cpu()
        if cav_dim == 1:
            features_batch = features_batch if features_batch.dim() == 2 else features_batch.flatten(start_dim=2).max(2).values
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

def _compute_cav(vecs: np.ndarray, targets: np.ndarray, cav_type: str = "svm"):
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
        scaler = np.mean(X**2)**0.5
        X /= scaler

    if "svm" in cav_type:
        linear = LinearSVC(random_state=0, fit_intercept=True)
        grid_search = GridSearchCV(linear, param_grid={"C": [10 ** i for i in range(-5, 5)]})
        grid_search.fit(X, targets, sample_weight=weights)
        linear = grid_search.best_estimator_
        print("Best C value:", grid_search.best_params_["C"])
        w = torch.Tensor(linear.coef_)

    elif "ridge" in cav_type:
        clf = Ridge(fit_intercept=True)
        grid_search = GridSearchCV(clf, param_grid={"alpha": [10 ** i for i in range(-5, 5)]})
        grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
        clf = grid_search.best_estimator_
        w = torch.tensor(clf.coef_)[None]

    elif "lasso" in cav_type:
        from sklearn.linear_model import Lasso
        clf = Lasso(fit_intercept=True)
        alphas = [10 ** i for i in range(-5, 1)]
        while True:

            grid_search = GridSearchCV(clf, param_grid={"alpha": alphas})
            grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
            w = torch.tensor(grid_search.best_estimator_.coef_)[None]
            if torch.sqrt((w ** 2).sum()) != 0:
                break
            else:
                alphas = alphas[:-1]
                if len(alphas) == 0:
                    raise ValueError("Lasso cannot be fit with given alphas.")
            

    elif "logistic" in cav_type:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(fit_intercept=True)
        grid_search = GridSearchCV(clf, param_grid={"C": [10 ** i for i in range(-5, 5)]})
        grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
        clf = grid_search.best_estimator_
        w = torch.tensor(clf.coef_)

    elif "signal" in cav_type:
        y = targets
        mean_y = y.mean()
        X_residuals = X - X.mean(axis=0)[None]
        covar = (X_residuals * (y - mean_y)[:, np.newaxis]).sum(axis=0) / (y.shape[0] - 1)
        vary = np.sum((y - mean_y) ** 2, axis=0) / (y.shape[0] - 1)
        w = (covar / vary)
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

    cav = w / torch.sqrt((w ** 2).sum())

    return cav


def compute_cavs(vecs: torch.Tensor, targets: torch.Tensor, type: str = "pattern_cav", normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes all concept activation vectors (CAVs) together for a set of vectors and targets.

    :param vecs:    torch.Tensor of shape (n_samples, n_features)
    :param targets: torch.Tensor of shape (n_samples, n_concepts)
    :param type:    str, type of CAV to compute. One of ["pattern_cav", "multi_cav"]
    :param normalize: bool, whether to normalize the CAVs to unit length
    :return:        tuple of (cavs, bias)
    """
    mu_x = vecs.mean(dim=0) 
    mu_y = targets.mean(dim=0)  
    X_centered = vecs - mu_x.unsqueeze(0)                           # (n_samples, n_features)
    Y_centered = targets - mu_y.unsqueeze(0)                        # (n_samples, n_concepts)
    n_samples = vecs.shape[0]

    if type == "pattern_cav":
        covars = (Y_centered.T @ X_centered) / (n_samples - 1)      # (n_concepts, n_features)
        vars = torch.sum(Y_centered**2, dim=0) / (n_samples - 1)    # (n_concepts,)
        inv_vars = torch.zeros_like(vars)
        nz = vars > 0
        inv_vars[nz] = 1.0 / vars[nz]
        cavs = covars * inv_vars.unsqueeze(1)                       # (n_concepts, n_features)

        if normalize:
            cavs /= torch.norm(cavs, dim=1, keepdim=True)
        
        # For zero-variance rows, best constant fit is mu_x
        bias = mu_x.unsqueeze(0) - mu_y.unsqueeze(1) * cavs         # (n_concepts, n_features)
        if (~nz).any():
            bias[(~nz), :] = mu_x.unsqueeze(0)

    elif type == "multi_cav":
        y_aug = Y_centered.T @ Y_centered
        cavs = torch.linalg.pinv(y_aug) @ (Y_centered.T @ X_centered)   # (n_concepts, n_features)
        if normalize:
            cavs /= torch.norm(cavs, dim=1, keepdim=True)
        bias = (mu_x - cavs.T @ mu_y).unsqueeze(0)                      # (1, n_features)

    else:
        raise KeyError(f"Unknown CAV type: {type}")

    return cavs, bias
