import logging
import functools
import pathlib
import sys

import numpy as np

from . import discrete
from . import raster
from . import trace
from . import q2bsp
from . import wal


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


def light_bsp(bsp: q2bsp.Q2Bsp, game_dir: pathlib.Path,
              faces: list[q2bsp.Face]) -> tuple[np.ndarray, np.ndarray]:
    tris = _parse_tris(faces)
    light_origin = np.array(
        next(iter(e['origin']
                  for e in bsp.entities
                  if e['classname'] == 'light'))
    )

    logger.info('calculate tc matrices')
    tex_vecs, face_idxs = _calculate_tex_vecs(faces)
    assert len(tex_vecs) == np.max(face_idxs) + 1

    logger.info('making source images')
    source_ims = _make_source_ims(game_dir, faces)

    logger.info('making source cdf')
    source_entries, source_cdf, _ = discrete.build_source_cdf(faces, source_ims)

    logger.info('tracing')
    lm_shapes = np.stack([face.lightmap_shape for face in faces], axis=0)
    lm_offsets = np.array([face.lightmap_offset for face in faces])
    output, counts = trace.trace(tris, light_origin, source_entries, source_cdf,
                                 face_idxs, tex_vecs, lm_shapes, lm_offsets)

    return output, counts


@functools.lru_cache(None)
def _luxel_area(face: q2bsp.Face) -> np.ndarray:
    return raster.render_aa_poly(face.lightmap_tcs, face.lightmap_shape)


def _create_new_lms(faces: list[q2bsp.Face], output: np.ndarray) \
        -> dict[q2bsp.Face, np.ndarray]:
    # Pull out each face's lightmap data.
    new_lms = {}
    for face in faces:
        offs = face.lightmap_offset
        h, w = face.lightmap_shape
        new_lm = output[offs:offs + w*h].reshape(h, w)

        # Texels that are only partially visible due to being on the edge of
        # the face should be boosted in brightness.
        new_lm = new_lm / np.maximum(_luxel_area(face), 1e-5)

        new_lms[face] = new_lm

    # Adjust levels to be sensible, and set the array to the correct format.
    max_count = np.max([np.max(new_lm) for new_lm in new_lms.values()])
    new_lms = {face: (255 * (new_lm / max_count) ** 0.5).astype(np.uint8)
               for face, new_lm in new_lms.items()}
    new_lms = {face: np.stack([new_lm] * 3, axis=2)
               for face, new_lm in new_lms.items()}

    return new_lms


def _read_pcx_palette(file):
    file.seek(-769, 2)
    if file.read(1) != b'\x0c':
        raise ValueError("Palette not found in PCX file")

    palette_data = file.read(768)
    palette = np.frombuffer(palette_data, dtype=np.uint8).reshape(-1, 3)
    return palette


def _make_source_ims(game_dir: pathlib.Path, faces: list[q2bsp.Face]) \
        -> dict[q2bsp.Face, np.ndarray]:
    # Load the palette.
    with (game_dir / 'pics' / 'colormap.pcx').open('rb') as f:
        pal = _read_pcx_palette(f)

    # Load all texture images.
    logger.info('loading textures')
    textures: dict[str, np.ndarray] = {}
    for face in faces:
        tex_name = face.tex_info.texture
        if tex_name not in textures:
            # Only .wal supported at the moment.
            with (game_dir / 'textures' / f'{tex_name}.wal').open('rb') as f:
                textures[tex_name] = wal.read_wal(f).images[0]

    # Initialize source ims as the lightmap.
    logger.info('extracting lightmaps as initial source')
    source_ims: dict[q2bsp.Face, np.ndarray] = {}
    for face in faces:
        source_ims[face] = face.extract_lightmap(0) / 255.

    # Make luxels that do not appear on the face black.
    logger.info('scaling sources by face intersection')
    for face in faces:
        source_ims[face] *= _luxel_area(face)[:, :, None]

    # Scale by texture color.
    logger.info('scaling sources by texture color')
    for face in faces:
        # Create an image where each pixel corresponds with the reflectivity of
        # the corresponding luxel.
        mins = np.floor(
            np.min(np.array(list(face.tex_coords)), axis=0).astype(np.float32)
                / 16
        ).astype(np.int32)
        tex_im = textures[face.tex_info.texture]
        lm_shape = face.lightmap_shape
        s = (np.arange(lm_shape[1] * 16) - 8 + mins[0] * 16) % tex_im.shape[1]
        t = (np.arange(lm_shape[0] * 16) - 8 + mins[1] * 16) % tex_im.shape[0]
        tex_im_remapped_16x = tex_im[t[:, None], s[None, :]]

        tex_im_remapped = np.mean(
            pal[tex_im_remapped_16x].reshape(
                (lm_shape[0], 16, lm_shape[1], 16, 3)
            ) / 255.,
            axis=(1, 3)
        )

        # Scale the source map by this reflectivity.
        source_ims[face] *= tex_im_remapped

    for face in faces:
        source_ims[face] = (source_ims[face] * 255.).astype(np.uint8)

    return source_ims


def main():
    logging.basicConfig(level=logging.DEBUG)

    game_dir = pathlib.Path(sys.argv[1])
    bsp_in_fname = sys.argv[2]
    if len(sys.argv) > 3:
        bsp_out_fname = sys.argv[3]
    else:
        bsp_out_fname = None

    logger.info(f'loading bsp {bsp_in_fname}')
    with open(bsp_in_fname, 'rb') as f:
        bsp = q2bsp.Q2Bsp(f)

    faces = [face for face in bsp.faces if face.has_lightmap(0)]

    logger.info('tracing')
    output, counts = light_bsp(bsp, game_dir, faces)
    print(repr(counts))

    if bsp_out_fname is not None:
        logger.info(f'writing bsp {bsp_out_fname}')
        new_lms = _create_new_lms(faces, output)

        # Rewrite the lightmap lump.
        with (open(bsp_in_fname, 'rb') as in_f, open(bsp_out_fname, 'wb') as out_f):
            q2bsp.rewrite_lightmap(in_f, new_lms, out_f)

if __name__ == "__main__":
    main()
