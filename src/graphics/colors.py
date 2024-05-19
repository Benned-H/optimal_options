"""This module provides utility functions for working with color values."""

import numpy as np
from matplotlib import cm


def equally_spaced_colors(n: int):
    """Create N equally spaced RGB color tuples.

    :param      n       Number of colors to generate
    :returns    List of N equally spaced (r, g, b) color tuples
    """
    color_spacing = np.linspace(0.0, 1.0, num=n, endpoint=False)
    hsv_map = cm.get_cmap("hsv")
    colors = [hsv_map(color, bytes=True) for color in color_spacing]

    return colors
