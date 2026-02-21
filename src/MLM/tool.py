import numpy as np


def load_layer_basis(vaspfile):
    b1 = np.loadtxt(vaspfile, skiprows=2, max_rows=1)[:2]
    b2 = np.loadtxt(vaspfile, skiprows=3, max_rows=1)[:2]
    B = np.array([b1, b2]).T
    return B


def angle_in_degrees(vec1, vec2):
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    dot = v1 @ v2
    dot = np.clip(
        dot, -1, 1
    )  # sometimes it goes above 1 or below -1, need to clip for np.acos to work
    rad = np.acos(dot)
    return np.degrees(rad)


