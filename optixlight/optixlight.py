import logging
import sys

import numpy as np

from pyquake import boxpack
from . import trace
from . import q2bsp


logger = logging.getLogger(__name__)
_ATLAS_SHAPE = (256, 256)


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


def _make_atlas(bsp: q2bsp.Q2Bsp) -> boxpack.BoxPacker:
    lightmap_shapes = {
        face: face.lightmap_shape for face in bsp.faces if
        face.has_lightmap(0)
    }
    lightmap_shapes = dict(reversed(sorted(
        lightmap_shapes.items(), key=lambda x: x[1][0] * x[1][1]
    )))
    box_packer = boxpack.BoxPacker(_ATLAS_SHAPE)
    for face, lightmap_shape in lightmap_shapes.items():
        if not box_packer.insert(face, lightmap_shape):
            raise Exception('lightmap too small')

    return box_packer


def _calculate_tex_vecs(box_packer: boxpack.BoxPacker, bsp: q2bsp.Q2Bsp) \
        -> tuple[np.ndarray, np.ndarray]:
    face_to_mat = {}
    for face, (r, c) in box_packer:
        tex_coords = np.array(list(face.tex_coords))
        mins = np.floor(
            np.min(tex_coords, axis=0).astype(np.float32) / 16
        ).astype(np.int32)

        M = np.empty((2, 4))
        M[0, :3] = np.array(face.tex_info.vec_s) / 16
        M[1, :3] = np.array(face.tex_info.vec_t) / 16
        M[0, 3] = face.tex_info.dist_s / 16 + 0.5 - mins[0] + c
        M[1, 3] = face.tex_info.dist_t / 16 + 0.5 - mins[1] + r

        face_to_mat[face] = M

    face_idxs = np.repeat(np.arange(len(bsp.faces)),
                          [face.num_edges - 2 for face in bsp.faces])

    return np.stack([
        face_to_mat[face]
        for face in bsp.faces if face.has_lightmap
    ], axis=0), face_idxs


def light_bsp(bsp: q2bsp.Q2Bsp) -> tuple[boxpack.BoxPacker, np.ndarray,
        np.ndarray]:
    tris = _parse_tris(bsp)
    light_origin = np.array(
        next(iter(e['origin']
                  for e in bsp.entities
                  if e['classname'] == 'light'))
    )

    logger.info('calculate tc matrices')
    box_packer = _make_atlas(bsp)
    tex_vecs, face_idxs = _calculate_tex_vecs(box_packer, bsp)
    assert len(tex_vecs) == np.max(face_idxs) + 1

    logger.info('tracing')
    output, counts = trace.trace(tris, light_origin, face_idxs, tex_vecs,
                                 _ATLAS_SHAPE)

    return box_packer, output, counts


def main():
    logging.basicConfig(level=logging.DEBUG)

    bsp_fname = sys.argv[1]

    logger.info(f'loading bsp {bsp_fname}')
    with open(bsp_fname, 'rb') as f:
        bsp = q2bsp.Q2Bsp(f)
    box_packer, output, counts = light_bsp(bsp)
    print(repr(counts))


if __name__ == "__main__":
    main()
