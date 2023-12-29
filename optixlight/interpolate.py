import collections
import logging

import numpy as np
import scipy

from . import q2bsp


logger = logging.getLogger(__name__)


def _build_face_groups(faces: list[q2bsp.Face]) -> list[list[q2bsp.Face]]:
    face_groups_dict = collections.defaultdict(list)
    for face in faces:
        face_groups_dict[(face.plane_id, face.plane_back)].append(face)

    return list(face_groups_dict.values())


def _augment(a):
    out = np.ones(a.shape[:-1] + (a.shape[-1] + 1,), dtype=a.dtype)
    out[..., :-1] = a
    return out


def _interpolate_face_group(
    face_group: list[q2bsp.Face],
    lms: dict[q2bsp.Face, np.ndarray],
    world_to_tcs: dict[q2bsp.Face, np.ndarray],
    tc_to_worlds: dict[q2bsp.Face, np.ndarray],
    areas: dict[q2bsp.Face, np.ndarray],
    coms: dict[q2bsp.Face, np.ndarray],
) -> dict[q2bsp.Face,  np.ndarray]:

    # Make arrays of points and values for which we already have data.
    points_list = []
    values_list = []
    for face in face_group:
        # TODO: Merge samples with a small area so that each sample has a
        # minimum total area.  Without this small faces are likely to be noisy.
        mask = areas[face] > 0.0
        points_list.append(
            _augment(coms[face][mask])
            @ tc_to_worlds[face].T
            @ world_to_tcs[face_group[0]][:, :3].T
        )
        values_list.append(lms[face][mask])
    points = np.concatenate(points_list)
    values = np.concatenate(values_list)

    if len(points) == 0:
        # Some faces contain only degenerate faces with zero area... in this
        # case just return their lightmaps unmodified.
        total_area = sum(np.sum(areas[face]) for face in face_group)
        if total_area == 0:
            logger.warning('not interpolating face group with %.5f area and %s'
                           ' faces', total_area, len(face_group))
            return {face: lms[face] for face in face_group}

    # Make an array of points for which we want data.
    xi_list = []
    idxs_list = []
    n = 0
    for face in face_group:
        shape = face.lightmap_shape
        xi_list.append(
            _augment(
                np.stack(np.meshgrid(np.arange(shape[1]) + 0.5,
                                     np.arange(shape[0]) + 0.5,
                                     indexing='xy'),
                         axis=-1).reshape(-1, 2)
            )
            @ tc_to_worlds[face].T
            @ world_to_tcs[face_group[0]][:, :3].T
        )
        idxs_list.append(n + np.arange(np.prod(shape)).reshape(shape))
        assert np.prod(shape) == len(xi_list[-1])
        n += len(xi_list[-1])
    xi = np.concatenate(xi_list)
    assert n == len(xi)

    # Interpolate / extrapolate.
    yi = scipy.interpolate.griddata(points, values, xi, method='nearest')

    # Pack up the lightmaps.
    return {
        face: yi[idx]
        for idx, face in zip(idxs_list, face_group)
    }


def interpolate_lightmaps(
    faces: list[q2bsp.Face],
    lms: dict[q2bsp.Face, np.ndarray],
    world_to_tcs: dict[q2bsp.Face, np.ndarray],
    tc_to_worlds: dict[q2bsp.Face, np.ndarray],
    areas: dict[q2bsp.Face, np.ndarray],
    coms: dict[q2bsp.Face, np.ndarray],
) -> dict[q2bsp.Face,  np.ndarray]:
    """
    Fix edges by interpolating lightmaps between adjacent faces.

    Arguments:
        faces: Faces whose lightmaps are to be interpolated.
        lms: Lightmaps to be interpolated.
        world_to_tcs: World to texture coord mappings for each face.
        tc_to_worlds: Texture coord to world mappings for each face.
        areas: Area of each luxel.
        com: Centre-of-mass of each luxel.

    Returns:
        The interpolated lightmaps.
    """

    out_lms = {}
    _build_face_groups(faces)
    for face_group in _build_face_groups(faces):
        out_lms.update(
            _interpolate_face_group(
                face_group,
                lms,
                world_to_tcs,
                tc_to_worlds,
                areas,
                coms,
            )
        )

    return out_lms
