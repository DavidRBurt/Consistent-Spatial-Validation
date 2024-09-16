from typing import List, Tuple
import numpy as np
from numpy.typing import ArrayLike
import queue


def make_split(points: ArrayLike, dim: int) -> Tuple[bool, ArrayLike, ArrayLike]:
    """
    Splits points along dimension d of the second index in the array based on whether points are on left or right half
    of [0, 1] on this dimension.

    Returns: Boolean indicating if both halves contain points, points on left half, rescaled by a factor of 2
    points on right half, shifted and rescaled by a factor of 2.
    """
    n, d = points.shape

    assert np.max(points) <= 1.0
    assert np.min(points) >= 0.0

    assert dim < d

    left_half_inds = np.where(points[:, dim] < 0.5)[0]
    # Check if either half is empty
    half_is_empty = len(left_half_inds) in [0, n]
    # Split array into both halves, rescale/shift points to make new boxes still [0, 1]
    left_half = points[left_half_inds]
    left_half[:, dim] = 2 * left_half[:, dim]
    right_half = np.delete(points, left_half_inds, axis=0)
    right_half[:, dim] = 2 * (right_half[:, dim] - 0.5)
    return half_is_empty, left_half, right_half


def approximate_fill_distance(points: ArrayLike) -> float:
    """
    Given an n x d numpy array of points, approximates fill distance of points in [0, 1]^d.
    Precisely, returns an r such that,

    fill distance / sqrt(d) <= r <= 4 * fill distance
    """
    # Ensure points are in [0, 1]^d
    assert np.max(points) <= 1.0
    assert np.min(points) >= 0

    dimension = points.shape[-1]

    # If no points, break immediately
    half_is_empty = points.shape[0] == 0
    # Haven't split any times
    num_splits = 0
    # Use a queue to store partitions of data points
    all_pts = queue.Queue()
    # Essentially this is the root of a tree. store its depth as 0
    all_pts.put((points, num_splits))
    # While things aren't empty
    while not half_is_empty:
        # Get points at current leaf, as well as depth of leaf
        cur_points, num_splits = all_pts.get()
        # Split the leaf into two children based on which half each point is in on the dimension
        # cycle over dimension with modular arithmetic
        half_is_empty, left_half, right_half = make_split(
            cur_points, num_splits % dimension
        )
        # Add children to tree, store their depth
        all_pts.put((left_half, num_splits + 1))
        all_pts.put((right_half, num_splits + 1))

    # Depth of deepest level which was complete, divided by dimension is number of times each dim was split
    # 2 to this power is side length of cubes such that we can partition cube into cubes of that size and they
    # are all non-empty. This is closely related to fill distance
    return 2 ** (-np.floor(num_splits / dimension))

