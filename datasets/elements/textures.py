import numpy as np

from .shapes import circle


def solid(concept, **kwargs):
    return concept


def random_spots(concept, spot_size=1, seed=None):
    n_spots = len(concept) * 3 // spot_size**2
    allowed_indicies = np.where(concept == 255)
    allowed_indicies = list(zip(*allowed_indicies))
    rng = np.random.default_rng(seed)
    spot_indices = rng.choice(allowed_indicies, n_spots, replace=False)
    spot_indices = spot_indices.T
    concept[spot_indices[0], spot_indices[1]] = 0
    return concept


def regular_spots(concept, spot_frac=0.1, seed=None):
    size = len(concept)
    spot_size = int(size * spot_frac)
    spot = np.pad(circle(spot_size), spot_size // 2).astype(float) / 255

    # Create extras so we can shift the texture
    n_spots = size // len(spot) + 4

    # Combine spots to make regular pattern
    tiled_spots = np.tile(spot, (n_spots, n_spots))

    # Shift the texture so that it applies differently each time
    rng = np.random.default_rng(seed)
    x_shift, y_shift = (rng.random(2) * spot_size * 2).astype(int)
    tiled_spots = tiled_spots[x_shift : size + x_shift, y_shift : size + y_shift]

    # Combine with the shape
    concept = (1 - tiled_spots) * concept
    concept = concept.astype(np.uint8)
    return concept


def polka(concept, spot_frac=0.09, seed=None):
    size = len(concept)
    spot_size = int(size * spot_frac)
    spot = np.pad(circle(spot_size), spot_size // 2).astype(float) / 255

    # Create extras so we can shift the texture
    n_spots = size // len(spot) + 4
    # Combine multiple spots/white space to create polka pattern
    tiled_spots = np.concatenate([spot, np.zeros_like(spot)])
    tiled_spots = np.concatenate((tiled_spots, tiled_spots[::-1, :]), axis=1)
    tiled_spots = np.tile(tiled_spots, (n_spots // 2, n_spots // 2))

    # Shift the texture so that it applies differently each time
    rng = np.random.default_rng(seed)
    x_shift, y_shift = (rng.random(2) * spot_size * 2).astype(int)
    tiled_spots = tiled_spots[x_shift : size + x_shift, y_shift : size + y_shift]

    # Combine with the shape
    concept = (1 - tiled_spots) * concept
    concept = concept.astype(np.uint8)
    return concept


def chequerboard(concept, width_frac=0.1, seed=None):
    size = len(concept)

    # Increase size so we can shift the texture by at least one chequer
    canvas_size = int(size * (1 + width_frac * 2))
    canvas_width_frac = size / canvas_size * width_frac

    # Create chequerboard
    x, y = np.meshgrid(np.linspace(0, 1, canvas_size), np.linspace(0, 1, canvas_size))
    texture = np.mod(np.floor(x // canvas_width_frac + y // canvas_width_frac), 2)

    # Shift the texture so that it applies differently each time
    rng = np.random.default_rng(seed)
    x_shift, y_shift = (rng.random(2) * size * width_frac).astype(int)
    texture = texture[x_shift : size + x_shift, y_shift : size + y_shift]

    # Combine with the shape
    concept = texture * concept
    concept = concept.astype(np.uint8)
    return concept


def create_stripe_texture(size, width_frac=0.1, direction="horizontal", seed=None):
    # Increase size so we can shift the texture by at least one chequer
    canvas_size = int(size * (1 + width_frac * 4))
    canvas_width_frac = size / canvas_size * width_frac
    x, y = np.meshgrid(np.linspace(0, 1, canvas_size), np.linspace(0, 1, canvas_size))

    if direction == "horizontal":
        mask = np.mod(y, canvas_width_frac * 2) < canvas_width_frac
    elif direction == "vertical":
        mask = np.mod(x, canvas_width_frac * 2) < canvas_width_frac
    elif direction == "diagonal":
        mask = np.mod(y + x, canvas_width_frac * 2) < canvas_width_frac
    elif direction == "diagonal_alt":
        mask = np.mod(y - x, canvas_width_frac * 2) < canvas_width_frac
    else:
        raise ValueError(
            "Invalid direction parameter. Must be one of 'horizontal', 'vertical', 'diagonal', or 'diagonal_alt'."
        )

    texture = np.where(mask, 1, 0)

    # Shift the texture so that it applies differently each time
    rng = np.random.default_rng(seed)
    x_shift, y_shift = (rng.random(2) * 4 * size * width_frac).astype(int)
    texture = texture[x_shift : size + x_shift, y_shift : size + y_shift]

    return texture


def striped(concept, width_frac=0.1, direction="horizontal", seed=None):
    size = len(concept)
    texture = create_stripe_texture(size, width_frac, direction, seed)
    concept = texture * concept
    concept = concept.astype(np.uint8)
    return concept


def striped_horizontal(concept, **kwargs):
    return striped(concept, direction="horizontal", **kwargs)


def striped_vertical(concept, **kwargs):
    return striped(concept, direction="vertical", **kwargs)


def striped_diagonal(concept, **kwargs):
    return striped(concept, direction="diagonal", **kwargs)


def striped_diagonal_alt(concept, **kwargs):
    return striped(concept, direction="diagonal_alt", **kwargs)
