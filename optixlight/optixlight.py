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
        # Note a few percent of faces in Quake 2 maps are not convex. This
        # triangulation will be slightly wrong for those but I doubt it'll be a
        # big deal.
        for v2, v3 in zip(vertices[1:-1], vertices[2:]):
            tris.append([v1, v2, v3])
    return np.array(tris)


def _calculate_tex_vecs(faces: list[q2bsp.Face]) \
        -> tuple[np.ndarray, np.ndarray]:
    world_to_tcs = []
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

        world_to_tcs.append(M)

    face_idxs = np.repeat(np.arange(len(world_to_tcs)),
                          [(face.num_edges - 2) for face in faces])

    return np.stack(world_to_tcs, axis=0), face_idxs


def _find_true_normal(face):
    # Find vector that points out from the front of a face.
    pts = np.array(list(face.vertices))
    diffs = pts[1:] - pts[0]
    normal = np.sum(np.cross(diffs[1:], diffs[:-1]), axis=0)
    normal /= np.linalg.norm(normal)
    return normal


def _invert_tex_vecs(world_to_tcs, faces):
    """For each face, return a 3x3 map from TCs to world coords.

    Each element `M` satisfies `M @ [s, t, 1] = [x, y, z]`,  for a texture
    coordinate [s, t] and corresponding 3D point [x, y, z].

    Note that this function can return points behind the geometry, since in
    Quake 2 some faces do not actually have coplanar points!
    """

    face_verts = np.array([next(iter(face.vertices)) for face in faces])
    normals = np.array([face.plane.normal for face in faces])

    # For each face make a 4x4 matrix M that maps an augmented world space
    # point [x, y, z, 1], which is the face's plane, to [s, t, 0, 1].
    M = np.concatenate([
        world_to_tcs,
        np.concatenate([
            normals,
            -np.einsum('mi,mi->m', face_verts, normals)[:, None]
        ], axis=1)[:, None, :],
        np.broadcast_to(
            np.array([0, 0, 0, 1])[None, None, :],
            (len(world_to_tcs), 1, 4)
        )
    ], axis=1)

    # Invert and get rid of the last row and the 3rd column, so that
    # multiplying by [s, t, 1] gives [x, y, z], a point in world space.
    return np.linalg.inv(M)[:, :3, [0, 1, 3]]


def light_bsp(bsp: q2bsp.Q2Bsp, game_dir: pathlib.Path,
              faces: list[q2bsp.Face]) -> tuple[np.ndarray, np.ndarray]:
    tris = _parse_tris(faces)

    logger.info('calculate tc matrices')
    world_to_tcs, face_idxs = _calculate_tex_vecs(faces)
    assert len(world_to_tcs) == np.max(face_idxs) + 1
    tc_to_worlds = _invert_tex_vecs(world_to_tcs, faces)

    logger.info('making reflectivity images')
    reflectivity_ims = _make_reflectivity_ims(game_dir, faces)

    logger.info('making source images')
    source_ims = _make_source_ims(reflectivity_ims, faces)

    logger.info('making source cdf')
    source_entries, source_cdf, _ = discrete.build_source_cdf(faces, source_ims)

    logger.info('tracing')
    normals = np.stack([face.plane.normal for face in faces])
    true_normals = np.stack([_find_true_normal(face) for face in faces])
    flip_mask = np.einsum('ni,ni->n', normals, true_normals) < 0
    normals[flip_mask] *= -1

    lm_shapes = np.stack([face.lightmap_shape for face in faces], axis=0)
    lm_offsets = np.array([face.lightmap_offset for face in faces])
    output = trace.trace(tris, source_entries, source_cdf, reflectivity_ims,
                         face_idxs, normals, world_to_tcs, tc_to_worlds,
                         lm_shapes, lm_offsets)

    return output


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
        new_lm = output[offs:offs + w*h*3].reshape(h, w, 3)

        # Texels that are only partially visible due to being on the edge of
        # the face should be boosted in brightness.
        new_lm = new_lm / np.maximum(_luxel_area(face)[:, :, None], 1e-5)

        # Add in the direct light from the original lightmap.
        new_lm += face.extract_lightmap(0)

        new_lms[face] = new_lm

    # Adjust levels to be sensible, and set the array to the correct format.
    new_lms = {face: np.clip(new_lm, 0, 255).astype(np.uint8)
                for face, new_lm in new_lms.items()}

    return new_lms


def _read_pcx_palette(file):
    file.seek(-769, 2)
    if file.read(1) != b'\x0c':
        raise ValueError("Palette not found in PCX file")

    palette_data = file.read(768)
    palette = np.frombuffer(palette_data, dtype=np.uint8).reshape(-1, 3)
    return palette


def _make_reflectivity_ims(game_dir: pathlib.Path, faces: list[q2bsp.Face]) \
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

    # Scale by texture color.
    logger.info('scaling reflectivity by texture color')
    reflectivity_ims: list[np.ndarray] = []
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
        reflectivity_ims.append(tex_im_remapped)

    return reflectivity_ims


def _make_source_ims(reflectivity_ims: dict[q2bsp.Face, np.ndarray],
                     faces: list[q2bsp.Face]) -> dict[q2bsp.Face, np.ndarray]:
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
    logger.info('scaling sources by reflectivity')
    for reflectivity_im, face in zip(reflectivity_ims, faces, strict=True):
        source_ims[face] *= reflectivity_im

    # Make uint8.
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
    output = light_bsp(bsp, game_dir, faces)

    if bsp_out_fname is not None:
        logger.info(f'writing bsp {bsp_out_fname}')
        new_lms = _create_new_lms(faces, output)

        # Rewrite the lightmap lump.
        with (open(bsp_in_fname, 'rb') as in_f, open(bsp_out_fname, 'wb') as out_f):
            q2bsp.rewrite_lightmap(in_f, new_lms, out_f)

if __name__ == "__main__":
    main()
