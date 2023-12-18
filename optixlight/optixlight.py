import logging
import sys

import numpy as np

from . import raster
from . import trace
from . import q2bsp


logger = logging.getLogger(__name__)


def _parse_tris(faces: list[q2bsp.Face]) -> np.ndarray:
    # Break faces down into triangle fans.
    tris = []
    for f in faces:
        vertices = list(f.vertices)
        v1 = vertices[0]
        for v2, v3 in zip(vertices[1:-1], vertices[2:]):
            tris.append([v1, v2, v3])
    return np.array(tris)


def _calculate_tex_vecs(faces: list[q2bsp.Face]) \
        -> tuple[np.ndarray, np.ndarray]:
    tex_vecs = []
    for face in faces:
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

    face_idxs = np.repeat(np.arange(len(tex_vecs)),
                          [(face.num_edges - 2) for face in faces])

    return np.stack(tex_vecs, axis=0), face_idxs


def light_bsp(bsp: q2bsp.Q2Bsp, faces: list[q2bsp.Face]) -> tuple[np.ndarray, np.ndarray]:
    tris = _parse_tris(faces)
    light_origin = np.array(
        next(iter(e['origin']
                  for e in bsp.entities
                  if e['classname'] == 'light'))
    )

    logger.info('calculate tc matrices')
    tex_vecs, face_idxs = _calculate_tex_vecs(faces)
    assert len(tex_vecs) == np.max(face_idxs) + 1

    logger.info('tracing')
    lm_shapes = np.stack([face.lightmap_shape for face in faces], axis=0)
    lm_offsets = np.array([face.lightmap_offset for face in faces])
    output, counts = trace.trace(tris, light_origin, face_idxs, tex_vecs,
                                 lm_shapes, lm_offsets)

    return output, counts


def _texel_area(face: q2bsp.Face) -> np.ndarray:
    return raster.render_aa_poly(face.lightmap_tcs, face.lightmap_shape)


def _rewrite_bsp(faces: list[q2bsp.Face], output: np.ndarray, bsp_in_fname: str,
                 bsp_out_fname: str):
    # Pull out each face's lightmap data.
    new_lms = {}
    for face in faces:
        offs = face.lightmap_offset
        h, w = face.lightmap_shape
        new_lm = output[offs:offs + w*h].reshape(h, w)

        # Texels that are only partially visible due to being on the edge of
        # the face should be boosted in brightness.
        new_lm = new_lm / np.maximum(_texel_area(face), 1e-5)

        new_lms[face] = new_lm

    # Adjust levels to be sensible, and set the array to the correct format.
    max_count = np.max([np.max(new_lm) for new_lm in new_lms.values()])
    new_lms = {face: (256 * (new_lm / max_count) ** 0.5).astype(np.uint8)
               for face, new_lm in new_lms.items()}
    new_lms = {face: np.stack([new_lm] * 3, axis=2)
               for face, new_lm in new_lms.items()}

    # Rewrite the lightmap lump.
    with (open(bsp_in_fname, 'rb') as in_f, open(bsp_out_fname, 'wb') as out_f):
        q2bsp.rewrite_lightmap(in_f, new_lms, out_f)


def main():
    logging.basicConfig(level=logging.DEBUG)

    bsp_in_fname = sys.argv[1]
    if len(sys.argv) > 2:
        bsp_out_fname = sys.argv[2]
    else:
        bsp_out_fname = None

    logger.info(f'loading bsp {bsp_in_fname}')
    with open(bsp_in_fname, 'rb') as f:
        bsp = q2bsp.Q2Bsp(f)

    faces = [face for face in bsp.faces if face.has_lightmap(0)]

    logger.info('tracing')
    output, counts = light_bsp(bsp, faces)
    print(repr(counts))

    if bsp_out_fname is not None:
        logger.info(f'writing bsp {bsp_out_fname}')
        _rewrite_bsp(faces, output, bsp_in_fname, bsp_out_fname)


if __name__ == "__main__":
    main()
