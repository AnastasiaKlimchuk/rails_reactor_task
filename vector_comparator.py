import numpy as np

import config


def compare(a, b):
    b = b.astype(int)
    a = a.astype(int)
    norm_diff = np.linalg.norm(b - a)
    norm1 = np.linalg.norm(b)
    norm2 = np.linalg.norm(a)
    norm_distance = norm_diff / (norm1 + norm2)

    if norm_distance < config.threshold:
        return True
    else:
        return False

