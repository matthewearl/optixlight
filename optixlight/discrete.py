import logging

import numpy as np

from . import q2bsp


logger = logging.getLogger(__name__)

_entry_dtype = np.dtype([
    ('p', np.float32),
    ('face_idx', np.uint32),
    ('tc', np.uint8, 2),
    ('color', np.uint8, 3),
])


def _binary_search(cdf, v):
    """Search for a value in a sorted array.

    Returns the largest `i` st `cdf[i] <= v`, or -1 if `v < cdf[i]` for all `i`.
    """

    # invariant: cdf[lo] <= v < cdf[hi]
    lo, hi = -1, len(cdf)
    while lo != hi - 1:
        mid = (lo + hi)//2
        if v < cdf[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def random_sample(cdf):
    """Return entry number of a randomly selected entry."""
    v = np.random.randint(0, 1<<32, dtype=np.uint32)
    return _binary_search(cdf, v) + 1


def build_source_cdf(bsp: q2bsp.Q2Bsp):
    logger.info('making entries')
    num_entries = np.sum([
        np.sum(np.any(face.extract_lightmap(0) > 0, axis=2))
        for face in bsp.faces
        if face.has_lightmap(0)
    ])
    entries = np.empty(num_entries, dtype=_entry_dtype)
    i = 0
    for face_idx, face in enumerate(f for f in bsp.faces if f.has_lightmap(0)):
        face_lm = face.extract_lightmap(0)
        for t in range(face_lm.shape[0]):
            for s in range(face_lm.shape[1]):
                color = face_lm[t, s]
                if np.any(color > 0):
                    assert i < num_entries
                    entries[i]['face_idx'] = face_idx
                    entries[i]['tc'] = (s, t)
                    entries[i]['color'] = color
                    i += 1
    assert i == num_entries

    logger.info('making importance')
    importance = entries['color'].sum(dtype=np.uint16, axis=1)

    cdf = np.cumsum(importance, dtype=np.float64)
    cdf = cdf[:-1] / cdf[-1]
    cdf = np.ldexp(cdf, 32)
    assert np.all(cdf <= 0xffffffff)
    assert np.all(cdf >= 0)
    cdf = cdf.astype(np.uint32)

    entries['p'] = np.ldexp(
        np.diff(
            cdf.astype(np.uint64), prepend=0, append=(1<<32)
        ).astype(np.float32),
        -32
    )

    return entries, cdf, importance
