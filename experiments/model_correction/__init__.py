from experiments.model_correction.base_correction import Vanilla
from experiments.model_correction.clarc import AClarc, Clarc, PClarc, ReactivePClarc
from experiments.model_correction.evaluate_by_subset_attacked import evaluate_by_subset_attacked
from experiments.model_correction.evaluate_model_correction import evaluate_model_correction
from experiments.model_correction.evaluate_heatmaps import evaluate_concept_heatmaps
from experiments.model_correction.dir_model import get_dir_model


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
