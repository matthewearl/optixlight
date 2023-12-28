import collections

import numpy as np
import scipy

from . import q2bsp


def _build_face_groups(faces: list[q2bsp.Face]) -> list[list[q2bsp.Face]]:
    # Build a graph with faces as vertices, with a graph edge between each
    # coplanar pair of faces that share a geometry edge, and are also in the
    # same plane
    edge_to_faces: dict[int, list[q2bsp.Face | None]] = collections.defaultdict(
        lambda: [None, None]
    )
    for face in faces:
        for edge_id in face.signed_edge_indices:
            if edge_id < 0:
                edge_to_faces[-edge_id][0] = face
            else:
                edge_to_faces[edge_id][1] = face

    graph_edges: dict[q2bsp.Face, list[q2bsp.Face]] = (
        collections.defaultdict(list)
    )
    for face1, face2 in edge_to_faces.values():
        if (face1 is not None
                and face2 is not None
                and face1.plane_id == face2.plane_id
                and face.plane_back == face2.plane_back):
            graph_edges[face1].append(face2)
            graph_edges[face2].append(face1)

    # Find connected subgraphs
    face_to_group: dict[q2bsp.Face, int] = {}
    def flood_fill(face: q2bsp.Face,
                   group_num: int,
                   face_to_group: dict[q2bsp.Face, int]):
        """Assign a group num to this face's connected subgraph."""
        assert face not in face_to_group
        face_to_group[face] = group_num
        for neighbour in graph_edges[face]:
            if neighbour in face_to_group:
                assert face_to_group[neighbour] == group_num
            else:
                flood_fill(neighbour, group_num, face_to_group)

    num_groups = 0
    for face in faces:
        if face not in face_to_group:
            flood_fill(face, num_groups, face_to_group)
            num_groups += 1

    # Invert the dict into a list of face groups.
    face_groups: list[list[q2bsp.Face]] = [[] for _ in range(num_groups)]
    for face, group_num in face_to_group.items():
        face_groups[group_num].append(face)

    return face_groups


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
        mask = areas[face] > 0.2
        points_list.append(
            _augment(coms[face][mask])
            @ tc_to_worlds[face].T
            @ world_to_tcs[face_group[0], :, :3].T
        )
        values_list.append(lms[face][mask])
    points = np.concatenate(points_list)
    values = np.concatenate(values_list)

    # Make an array of points for which we want data.
    xi_list = []
    idxs_list = []
    n = 0
    for face in face_group:
        shape = face.lightmap_shape
        xi_list.append(
            _augment((np.indices(shape) + 0.5).transpose().reshape(-1, 2))
            @ tc_to_worlds[face].T
            @ world_to_tcs[face_group[0], :, :3].T
        )
        idxs_list.append(n + np.arange(np.prod(shape)).reshape(shape))
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
