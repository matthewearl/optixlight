import logging
import sys

import numpy as np

from . import trace
from . import q2bsp


logger = logging.getLogger(__name__)


def _parse_tris(bsp: q2bsp.Q2Bsp) -> np.ndarray:
    # Break faces down into triangle fans.
    tris = []
    for f in bsp.faces:
        if f.has_lightmap(0):
            vertices = list(f.vertices)
            v1 = vertices[0]
            for v2, v3 in zip(vertices[1:-1], vertices[2:]):
                tris.append([v1, v2, v3])
    return np.array(tris)


def _calculate_tex_vecs(bsp: q2bsp.Q2Bsp) -> tuple[np.ndarray, np.ndarray]:
    tex_vecs = []
    for face in bsp.faces:
        if face.has_lightmap(0):
            tex_coords = np.array(list(face.tex_coords))
            mins = np.floor(
                np.min(tex_coords, axis=0).astype(np.float32) / 16
            ).astype(np.int32)

            M = np.empty((2, 4))
            M[0, :3] = np.array(face.tex_info.vec_s) / 16
            M[1, :3] = np.array(face.tex_info.vec_t) / 16
            M[0, 3] = face.tex_info.dist_s / 16 + 0.5 - mins[0]
            M[1, 3] = face.tex_info.dist_t / 16 + 0.5 - mins[1]

            tex_vecs.append(M)

    face_idxs = np.repeat(np.arange(len(bsp.faces)),
                          [(face.num_edges - 2) if face.has_lightmap(0) else 0
                           for face in bsp.faces])

    return np.stack(tex_vecs, axis=0), face_idxs


def light_bsp(bsp: q2bsp.Q2Bsp) -> tuple[np.ndarray, np.ndarray]:
    tris = _parse_tris(bsp)
    light_origin = np.array(
        next(iter(e['origin']
                  for e in bsp.entities
                  if e['classname'] == 'light'))
    )

    logger.info('calculate tc matrices')
    tex_vecs, face_idxs = _calculate_tex_vecs(bsp)
    assert len(tex_vecs) == np.max(face_idxs) + 1

    logger.info('tracing')
    lm_shapes = np.stack(
        [face.lightmap_shape for face in bsp.faces if face.has_lightmap(0)],
        axis=0
    )
    lm_offsets = np.array([
        face.lightmap_offset for face in bsp.faces if face.has_lightmap(0)
    ])
    output, counts = trace.trace(tris, light_origin, face_idxs, tex_vecs,
                                 lm_shapes, lm_offsets)

    return output, counts


def main():
    logging.basicConfig(level=logging.DEBUG)

    bsp_fname = sys.argv[1]

    logger.info(f'loading bsp {bsp_fname}')
    with open(bsp_fname, 'rb') as f:
        bsp = q2bsp.Q2Bsp(f)
    output, counts = light_bsp(bsp)
    print(repr(counts))


if __name__ == "__main__":
    main()
