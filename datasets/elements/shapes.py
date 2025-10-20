import numpy as np


def square(size):
    return (np.ones((size, size)) * 255).astype(np.uint8)


def plus(size, thickness_ratio=0.25):
    concept = np.zeros((size, size)).astype(np.uint8)
    centre = size // 2
    odd = size % 2 == 1
    thickness = int(thickness_ratio * size) // 2
    l = centre - thickness
    r = centre + thickness + odd
    concept[:, l:r] = 255
    concept[l:r, :] = 255
    return concept


def circle(size):
    radius = size // 2
    centre_float = size / 2 - 0.5
    x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
    concept = ((x_grid - centre_float) ** 2 + (y_grid - centre_float) ** 2) <= (radius) ** 2
    concept = concept.astype(np.uint8) * 255
    return concept


def triangle(size):
    concept = square(size)
    centre_float = size / 2 - 0.5
    x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
    top_left = (x_grid + y_grid * 0.5) < centre_float
    top_right = (x_grid - y_grid * 0.5) > centre_float
    concept[top_left] = 0
    concept[top_right] = 0
    return concept


def cross(size, thickness_ratio=0.25):
    concept = np.zeros((size, size)).astype(np.uint8)
    thickness = (size * thickness_ratio) // 2
    x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
    bl_tr_line = x_grid + y_grid
    tl_br_line = x_grid - y_grid
    bl_tr = (bl_tr_line + thickness > size - 1) & (bl_tr_line - thickness < size - 1)
    tl_br = (tl_br_line + thickness > 0) & (tl_br_line - thickness < 0)
    concept[bl_tr] = 255
    concept[tl_br] = 255
    return concept
