from experiments.clarc.base_correction import Vanilla
from experiments.clarc.clarc import AClarc, Clarc, PClarc, ReactivePClarc


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
