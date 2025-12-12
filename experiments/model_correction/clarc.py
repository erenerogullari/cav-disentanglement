import copy
import os

import h5py
import numpy as np
import torch
from zennit.core import stabilize
from experiments.model_correction.base_correction import LitClassifier, Freeze
from experiments.model_correction.base_correction import Vanilla
from pathlib import Path
from utils.cav import compute_cav
import logging 

log = logging.getLogger(__name__)

class Clarc(LitClassifier):
    def __init__(self, model, config, cav, **kwargs):
        super().__init__(model, config, **kwargs)

        self.std = None
        self.layer_name = config["layer_name"]
        self.dataset_name = config["dataset_name"]
        self.model_name = config["model_name"]
        self.mode = config["mode"]

        assert "artifact_sample_ids" in kwargs.keys(), "artifact_sample_ids have to be passed to ClArC correction methods"
        assert "sample_ids" in kwargs.keys(), "all sample_ids have to be passed to ClArC correction methods"

        self.artifact_sample_ids = kwargs["artifact_sample_ids"]
        self.sample_ids = kwargs["sample_ids"]
        self.classes = kwargs.get("classes", None)
        self.direction_mode = config["direction_mode"]

        log.info(f"Using {len(self.artifact_sample_ids)} artifact samples.")
        self.path = Path(config['dir_precomputed_data']) / f"{self.layer_name}.pth"

        self.cav = cav
        mean_length, mean_length_targets = self.compute_means(norm=False)
        self.mean_length = mean_length
        self.mean_length_targets = mean_length_targets
        hooks = []
        for n, m in self.model.named_modules():
            if n == self.layer_name:
                log.info("Registered forward hook.")
                hooks.append(m.register_forward_hook(self.clarc_hook))
        self.hooks = hooks

    def compute_means(self, norm=False):
        variables = torch.load(self.path)
        vecs = variables["encs"]
        labels = variables["labels"]

        # Only keep samples that are in self.sample_ids (usually training set)
        vecs = vecs[self.sample_ids].to(self.cav.device)
        labels = labels[self.sample_ids]

        target_ids = np.array(
            [np.argwhere(self.sample_ids == id_)[0][0] for id_ in self.artifact_sample_ids if
             np.argwhere(self.sample_ids == id_).any()])
        targets = np.array([1 * (j in target_ids) for j, x in enumerate(self.sample_ids)])
        mean_length = (vecs[targets == 0].flatten(start_dim=1)  * self.cav).sum(1).mean(0)
        mean_length_targets = (vecs[targets == 1].flatten(start_dim=1) * self.cav).sum(1).mean(0)

        return mean_length, mean_length_targets
        


    def clarc_hook(self, m, i, o):
        pass

    def configure_callbacks(self):
        return [Freeze(
            self.layer_name
        )]


class PClarc(Clarc):
    def __init__(self, model, config, cav, **kwargs):
        super().__init__(model, config, cav, **kwargs)

        if os.path.exists(self.path):
            self.cav = cav
            mean_length, mean_length_targets = self.compute_means(norm=False) 
            self.mean_length = mean_length
            self.mean_length_targets = mean_length_targets
        else:
            if self.hooks and not kwargs.get("eval_mode", False):
                for hook in self.hooks:
                    log.info("Removed hook. No hook should be active for training.")
                    hook.remove()
                self.hooks = []

    def clarc_hook(self, m, i, o):
        outs = o + 0
        dim_orig = 4
        if outs.dim() == 2:
            dim_orig = 2
            outs = outs[..., None, None]
        elif outs.dim() == 3:
            dim_orig = 3
            outs = outs[..., None]

        cav = self.cav.to(outs)
        if self.mode == "full":
            length = (outs.flatten(start_dim=1) * cav).sum(1)
        else:
            length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        v = self.cav.to(outs)
        dim = 0 if cav.dim() == 1 else 1
        beta = (cav * v).sum(dim)
        mag = (self.mean_length - length).to(outs) / stabilize(beta)
        v = v.reshape(1, *outs.shape[1:]) if self.mode == "full" else v[..., None, None]
        addition = (mag[:, None, None, None] * v)
        acts = outs + addition
        if dim_orig == 2:
            acts = acts.squeeze(-1).squeeze(-1) 
        elif dim_orig == 3:
            acts = acts.squeeze(-1)
        return acts

class ReactivePClarc(PClarc):
    def __init__(self, model, config, cav, **kwargs):
        super().__init__(model, config, cav, **kwargs)

        true_direction = self.config["direction_mode"]
        self.direction_mode = "svm"
        cav_svm, _, _ = self.compute_svm_cav(self.mode)
        self.cav_svm = cav_svm
        self.direction_mode = true_direction
        log.info("computed SVM-CAV for condition")

    def clarc_hook(self, m, i, o):
        outs = o + 0
        dim_orig = 4
        if outs.dim() == 2:
            dim_orig = 2
            outs = outs[..., None, None]
        elif outs.dim() == 3:
            dim_orig = 3
            outs = outs[..., None]
        # outs = outs[..., None, None] if is_2dim else outs
        cav = self.cav.to(outs)
        if self.mode == "full":
            length = (outs.flatten(start_dim=1) * cav).sum(1)
        else:
            length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        v = self.cav.to(outs)
        dim = 0 if cav.dim() == 1 else 1
        beta = (cav * v).sum(dim)
        mag = (self.mean_length - length).to(outs) / stabilize(beta)
        v = v.reshape(1, *outs.shape[1:]) if self.mode == "full" else v[..., None, None]
        addition = (mag[:, None, None, None] * v)

        ## implement condition
        contains_artifact = outs.flatten(start_dim=2).max(2).values @ self.cav_svm.T.to(outs.device) > 0
        addition = (contains_artifact[:, None, None] * addition)
        
        acts = outs + addition
        if dim_orig == 2:
            acts = acts.squeeze(-1).squeeze(-1) 
        elif dim_orig == 3:
            acts = acts.squeeze(-1)
        return acts
    
    def compute_svm_cav(self, mode):
        variables = torch.load(self.path)
        vecs = variables["encs"]
        labels = variables["labels"]

        # Only keep samples that are in self.sample_ids (usually training set)
        vecs = vecs[self.sample_ids]
        labels = labels[self.sample_ids]

        target_ids = np.array(
            [np.argwhere(self.sample_ids == id_)[0][0] for id_ in self.artifact_sample_ids if
             np.argwhere(self.sample_ids == id_).any()])
        targets = np.array([1 * (j in target_ids) for j, x in enumerate(self.sample_ids)])
        
        cav_svm, _ = compute_cav(vecs.numpy(), targets, cav_type="svm")

        return cav_svm, vecs, targets

class AClarc(Clarc):
    def __init__(self, model, config, cav, **kwargs):
        super().__init__(model, config, cav, **kwargs)
        self.lamb = self.config["lamb"] 

    def clarc_hook(self, m, i, o):
        outs = o + 0
        is_2dim = outs.dim() == 2
        outs = outs[..., None, None] if is_2dim else outs
        cav = self.cav.to(outs)
        if self.mode == "full":
            length = (outs.flatten(start_dim=1) * cav).sum(1)
        else:
            length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        v = self.cav.to(outs)
        dim = 0 if cav.dim() == 1 else 1
        beta = (cav * v).sum(dim)
        mag = (self.mean_length_targets - length).to(outs) / stabilize(beta)
        v = v.reshape(1, *outs.shape[1:]) if self.mode == "full" else v[..., None, None]

        addition = (mag[:, None, None, None] * v).requires_grad_()
        acts = outs + addition
        acts = acts.squeeze(-1).squeeze(-1) if is_2dim else acts
        return acts


def get_correction_method(method_name):
    CORRECTION_METHODS = {
        'Vanilla': Vanilla,
        'Clarc': Clarc,
        'AClarc': AClarc,
        'PClarc': PClarc,
        'ReactivePClarc': ReactivePClarc

    }

    assert method_name in CORRECTION_METHODS.keys(), f"Correction method '{method_name}' unknown," \
                                                     f" choose one of {list(CORRECTION_METHODS.keys())}"
    return CORRECTION_METHODS[method_name]