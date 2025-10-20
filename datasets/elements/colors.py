import numpy as np

COLORS = {
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "yellow": (1, 1, 0),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
}
COLORS = {k: np.array(v) for k, v in COLORS.items()}


def color_adjustment(seed=None, max_diff=0.4):
    rng = np.random.default_rng(seed)
    return 1 - rng.random() * max_diff
