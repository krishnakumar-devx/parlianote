"""Small statistics helpers shared by the app and the evaluation scripts."""

import numpy as np


def wilson(k: int, n: int, z: float = 1.96):
    """95% confidence interval for a proportion k/n (Wilson, 1927). More reliable than the textbook formula for small samples."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)
