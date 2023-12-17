import numpy as np


def _find_poi(pt1, pt2, d, axis):
    """Find the intersection of a line segment with a line.

    The line segment is defined as two points, and the line is parallel to one
    of the axes.
    """
    f = (d - pt1[axis]) / (pt2[axis] - pt1[axis])
    assert -1e-5 <= f <= 1+1e-5
    f = np.clip(f, 0, 1)
    poi = np.empty(2,)
    poi[axis] = d
    poi[1 - axis] = (1 - f) * pt1[1 - axis] + f * pt2[1 - axis]
    return poi


def _poly_truncate(pts, d, axis, side):
    """Truncate a polygon to an axially-aligned half-space"""
    keep = pts[:, axis] < d
    if side:
        keep = ~keep

    new_pts = []
    for pt1, pt2, keep1, keep2 in zip(pts, np.roll(pts, -1, axis=0),
                                      keep, np.roll(keep, -1),
                                      strict=True):
        if keep1:
            new_pts.append(pt1)
            if not keep2:
                new_pts.append(_find_poi(pt1, pt2, d, axis))
        elif keep2:
            new_pts.append(_find_poi(pt1, pt2, d, axis))
    if new_pts:
        return np.stack(new_pts)
    else:
        return np.empty((0, 2), dtype=pts.dtype)


def _poly_area(pts):
    """Return the area of polygon."""
    ds = pts[1:] - pts[0]
    return 0.5 * np.abs(np.sum(np.cross(ds[:-1], ds[1:])))


def _render_aa_poly_rec(pts, mins, maxs):
    """Rasterize a poly in some bbox.

    The poly is assumed to be clipped to the bbox."""

    if len(pts) == 0:
        # The poly does not intersect this subtree's rectangle at all.
        return np.zeros(np.flip(maxs - mins))

    area = _poly_area(pts)
    if np.allclose(np.product(maxs - mins), area):
        # The poly entirely encloses this subtree's rectangle.
        return np.ones(np.flip(maxs - mins))

    if mins[0] == maxs[0] - 1 and mins[1] == maxs[1] - 1:
        # Only one pixel in this subtree, so its colour should be the area of
        # the poly that intersects with this pixel.
        return np.array([[area]])

    # Otherwise, split along the longest axis, and recursively render each half.
    split_axis = np.argmax(maxs - mins)
    assert maxs[split_axis] - mins[split_axis] > 1
    d = (maxs[split_axis] + mins[split_axis]) // 2
    assert mins[split_axis] < d < maxs[split_axis]
    new_maxs = maxs.copy()
    new_maxs[split_axis] = d
    new_mins = mins.copy()
    new_mins[split_axis] = d

    return np.concatenate([
        _render_aa_poly_rec(_poly_truncate(pts, d, split_axis, False),
                            mins, new_maxs),
        _render_aa_poly_rec(_poly_truncate(pts, d, split_axis, True),
                            new_mins, maxs),
    ], axis=(1 - split_axis))


def render_aa_poly(pts: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Rasterize a polygon with accurate anti-aliasing.

    The value of each element in the output array indicates the area of the poly
    that intersects with the pixel.  The pixel corresponding with element
    `(i, j)` is a square `[j, j + 1] x [i, i + 1]`.

    Arguments:
        pts: (n, 2) array of points describing the polygon being rasterized.
        shape:  Shape of the output image.

    Returns:
        The rasterized polygon.
    """

    mins = np.array([0, 0], dtype=np.int32)
    maxs = np.array([shape[1], shape[0]], dtype=np.int32)
    for axis in range(2):
        pts = _poly_truncate(pts, mins[axis], axis, True)
        pts = _poly_truncate(pts, maxs[axis], axis, False)

    return _render_aa_poly_rec(pts, mins, maxs)
