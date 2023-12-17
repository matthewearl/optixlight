import logging

import numpy as np

from . import q2bsp


logger = logging.getLogger(__name__)

_leaf_dtype = np.dtype([
    ('face_idx', np.uint32),
    ('tc', np.uint8, 2),
    ('color', np.uint8, 3)
])


def _write_subtree(leaves: np.ndarray, importance: np.ndarray,
                   nodes: np.ndarray):
    n = len(leaves)
    assert n - 1 == len(nodes)
    if n > 1:
        n_left = (1 << int(np.ceil(np.log2(n)) - 1))
        assert n > n_left > 0

        imp_left = np.sum(importance[:n_left])
        imp_right = np.sum(importance[n_left:])
        nodes[0] = imp_left / (imp_left + imp_right)
        _write_subtree(leaves[:n_left], importance[:n_left], nodes[1:n_left])
        _write_subtree(leaves[n_left:], importance[n_left:], nodes[n_left:])


def random_sample(nodes):
    leaf_num = 0
    node_num = 0
    num_leaves = len(nodes) + 1
    n_left = (1 << int(np.ceil(np.log2(num_leaves)) - 1))

    while num_leaves > 1:
        if random.random() < nodes[node_num]:
            node_num += 1
            num_leaves = n_left
            n_left = n_left >> 1
        else:
            node_num += n_left
            num_leaves -= n_left
            leaf_num += n_left
            while n_left >= num_leaves:
                n_left = n_left >> 1
    assert num_leaves == 1
    return leaf_num


def build_probability_tree(bsp: q2bsp.Q2Bsp):
    logger.info('making leaves')
    num_leaves = np.sum([
        np.sum(np.any(face.extract_lightmap(0) > 0, axis=2))
        for face in bsp.faces
        if face.has_lightmap(0)
    ])
    leaves = np.empty(num_leaves, dtype=_leaf_dtype)
    i = 0
    for face_idx, face in enumerate(f for f in bsp.faces if f.has_lightmap(0)):
        face_lm = face.extract_lightmap(0)
        for t in range(face_lm.shape[0]):
            for s in range(face_lm.shape[1]):
                color = face_lm[t, s]
                if np.any(color > 0):
                    assert i < num_leaves
                    leaves[i]['face_idx'] = face_idx
                    leaves[i]['tc'] = (s, t)
                    leaves[i]['color'] = color
                    i += 1
    assert i == num_leaves

    logger.info('making importance')
    importance = leaves['color'].sum(dtype=np.uint16, axis=1)

    logger.info('making nodes')
    nodes = np.empty(len(leaves) - 1, dtype=np.float32)
    _write_subtree(leaves, importance, nodes)

    return leaves, nodes, importance
