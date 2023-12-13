import logging
import sys

import numpy as np

from . import trace
from . import q2bsp


def main():
    logging.basicConfig(level=logging.DEBUG)

    bsp_fname = sys.argv[1]

    with open(bsp_fname, 'rb') as f:
        bsp = q2bsp.Q2Bsp(f)

    # Break faces down into triangle fans.
    tris = []
    for f in bsp.faces:
        vertices = list(f.vertices)
        v1 = vertices[0]
        for v2, v3 in zip(vertices[1:-1], vertices[2:]):
            tris.append([v1, v2, v3])
    tris = np.array(tris)

    light_origin = np.array(
        next(iter(e['origin']
                  for e in bsp.entities
                  if e['classname'] == 'light'))
    )

    trace.trace(tris, light_origin, None, None)


if __name__ == "__main__":
    main()
