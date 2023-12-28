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
    for i1 in range(len(pts)):
        i2 = (i1 + 1) % len(pts)
        keep1 = keep[i1]
        keep2 = keep[i2]

        if keep1:
            pt1 = pts[i1]
            new_pts.append(pt1)
            if not keep2:
                pt2 = pts[i2]
                new_pts.append(_find_poi(pt1, pt2, d, axis))
        elif keep2:
            new_pts.append(_find_poi(pts[i1], pts[i2], d, axis))
    if new_pts:
        return np.stack(new_pts)
    else:
        return np.empty((0, 2), dtype=pts.dtype)


def _poly_area_and_com(pts):
    """Return the area of polygon."""
    ds = pts[1:] - pts[0]

    tri_area = 0.5 * np.cross(ds[:-1], ds[1:])
    tri_com = (ds[1:] + ds[:-1]) / 3.

    poly_area = np.sum(tri_area)
    poly_com = np.dot(tri_area, tri_com) / poly_area

    if poly_area < 0:
        poly_area = -poly_area

    poly_com += pts[0]

    return poly_area, poly_com


def _make_central_com(mins, maxs):
	return np.stack(
		np.broadcast_arrays(
			np.arange(mins[0], maxs[0])[None, :] + 0.5,
			np.arange(mins[1], maxs[1])[:, None] + 0.5,
		),
		axis=2
	)


def _render_aa_poly_rec(clip_mask, pts, mins, maxs):
    """Rasterize a poly in some bbox.

    The poly is assumed to be clipped to the bbox."""

    if len(pts) == 0:
        # The poly does not intersect this subtree's rectangle at all.
        return np.zeros(np.flip(maxs - mins)), _make_central_com(mins, maxs)

    area, com = None, None
    if clip_mask == 0xf:
        area, com = _poly_area_and_com(pts)
        if area > (maxs[0] - mins[0]) * (maxs[1] - mins[1]) * 0.999:
            # The poly entirely encloses this subtree's rectangle.
            return np.ones(np.flip(maxs - mins)), _make_central_com(mins, maxs)

    if mins[0] == maxs[0] - 1 and mins[1] == maxs[1] - 1:
        # Only one pixel in this subtree, so its colour should be the area of
        # the poly that intersects with this pixel.
        if area is None:
            area, com = _poly_area_and_com(pts)
        return np.array([[area]]), np.array([[com]])

    # Otherwise, split along the longest axis, and recursively render each half.
    split_axis = np.argmax(maxs - mins)
    assert maxs[split_axis] - mins[split_axis] > 1
    d = (maxs[split_axis] + mins[split_axis]) // 2
    assert mins[split_axis] < d < maxs[split_axis]
    new_maxs = maxs.copy()
    new_maxs[split_axis] = d
    new_mins = mins.copy()
    new_mins[split_axis] = d

    child1_areas, child1_coms = _render_aa_poly_rec(
        clip_mask | (1 << (split_axis << 1)),
        _poly_truncate(pts, d, split_axis, False),
        mins, new_maxs
    )
    child2_areas, child2_coms = _render_aa_poly_rec(
        clip_mask | (1 << ((split_axis << 1) | 1)),
        _poly_truncate(pts, d, split_axis, True),
        new_mins, maxs
    )

    return (
        np.concatenate([child1_areas, child2_areas], axis=(1 - split_axis)),
        np.concatenate([child1_coms, child2_coms], axis=(1 - split_axis)),
    )


def render_aa_poly(pts: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Rasterize a polygon with accurate anti-aliasing.

    The value of each element in the output array indicates the area of the poly
    that intersects with the pixel.  The pixel corresponding with element
    `(i, j)` is a square `[j, j + 1] x [i, i + 1]`.

    The function also returns the centre-of-mass for each output pixel.

    Arguments:
        pts: (n, 2) array of points describing the polygon being rasterized.
        shape:  Shape of the output image.

    Returns:
        The rasterized polygon `im`, and centres of mass `com`.  The shape of
        `im` is `shape`, and the centres of mass are `shape + (2,)`, with
        `com[i, j]` being the centre of mass for the pixel in the `i'th` row and
        the `j'th` column.
    """

    mins = np.array([0, 0], dtype=np.int32)
    maxs = np.array([shape[1], shape[0]], dtype=np.int32)
    for axis in range(2):
        pts = _poly_truncate(pts, mins[axis], axis, True)
        pts = _poly_truncate(pts, maxs[axis], axis, False)

    return _render_aa_poly_rec(0, pts, mins, maxs)
