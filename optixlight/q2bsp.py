import dataclasses
import enum
import functools
import itertools
import struct

from pyquake import ent

import numpy as np


class PlaneType(enum.Enum):
    AXIAL_X = 0
    AXIAL_Y = 1
    AXIAL_Z = 2
    NON_AXIAL_X = 3
    NON_AXIAL_Y = 4
    NON_AXIAL_Z = 5


@dataclasses.dataclass
class Plane:
    normal: tuple[float, float, float]
    dist: float
    plane_type: PlaneType

    def point_dist(self, point):
        return np.dot(point, self.normal) - self.dist

    def infront(self, point):
        return self.point_dist(point) >= 0


class TexInfoFlags(enum.IntFlag):
    LIGHT = 0x1
    SLICK = 0x2
    SKY = 0x4
    WARP = 0x8
    TRANS33 = 0x10
    TRANS66 = 0x20
    FLOWING = 0x40
    NODRAW = 0x80


@dataclasses.dataclass
class _DirEntry:
    offset: int
    size: int


@dataclasses.dataclass
class Face:
    bsp: 'Q2Bsp'
    plane_id: int
    plane_back: bool
    edge_list_idx: int
    num_edges: int
    texinfo_id: int
    styles: list[int]
    lightmap_offset: int

    @property
    @functools.lru_cache(None)
    def id_(self):
        return self.bsp.faces.index(self)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    @property
    def vert_indices(self):
        for edge_id in self.bsp.edge_list[self.edge_list_idx:self.edge_list_idx + self.num_edges]:
            if edge_id < 0:
                v = self.bsp.edges[-edge_id][1]
            else:
                v = self.bsp.edges[edge_id][0]
            yield v

    @property
    def signed_edge_indices(self):
        return self.bsp.edge_list[self.edge_list_idx:self.edge_list_idx + self.num_edges]

    @property
    def vertices(self):
        return (self.bsp.vertices[idx] for idx in self.vert_indices)

    @property
    def tex_coords(self):
        return [self.tex_info.vert_to_tex_coords(v) for v in self.vertices]

    @property
    def tex_info(self):
        return self.bsp.texinfo[self.texinfo_id]

    @property
    def plane(self):
        return self.bsp.planes[self.plane_id]

    @property
    @functools.lru_cache(None)
    def inverse_tc_matrix(self):
        plane = self.plane

        tex_info = self.tex_info
        A = np.stack([tex_info.vec_s, tex_info.vec_t, plane.normal])
        b = np.array([tex_info.dist_s, tex_info.dist_t, -plane.dist])

        # A @ vert + b = tex_coord
        return np.linalg.inv(A), b

    def tc_to_vert(self, tex_coord):
        augmented = np.zeros(tex_coord.shape[:-1] + (3,))
        augmented[..., :2] = tex_coord
        Ainv, b = self.inverse_tc_matrix
        return np.einsum('ij,...j->...i', Ainv, augmented - b)

    @property
    def lightmap_shape(self):
        tex_coords = np.array(list(self.tex_coords))

        mins = np.floor(np.min(tex_coords, axis=0).astype(np.float32) / 16).astype(np.int32)
        maxs = np.ceil(np.max(tex_coords, axis=0).astype(np.float32) / 16).astype(np.int32)

        size = (maxs - mins) + 1
        return (size[1], size[0])

    @property
    def lightmap_tcs(self):
        tex_coords = np.array(list(self.tex_coords))

        mins = np.floor(np.min(tex_coords, axis=0).astype(np.float32) / 16).astype(np.int32)
        maxs = np.ceil(np.max(tex_coords, axis=0).astype(np.float32) / 16).astype(np.int32)

        tex_coords -= mins * 16
        tex_coords += 8
        tex_coords /= 16.

        return tex_coords

    @property
    def has_any_lightmap(self):
        return any(self.has_lightmap(i) for i in range(4))

    def has_lightmap(self, lightmap_idx):
        return (
            self.tex_info.flags & (TexInfoFlags.SKY | TexInfoFlags.WARP) == 0
            and self.lightmap_offset != -1
            and self.styles[lightmap_idx] != 255
        )

    def extract_lightmap(self, lightmap_idx):
        assert self.has_lightmap(lightmap_idx)

        shape = self.lightmap_shape + (3,)
        size = shape[0] * shape[1] * 3

        idx = 0
        for i in range(lightmap_idx):
            if self.has_lightmap(i):
                idx += 1

        lightmap = np.array(list(
            self.bsp.lightmap[self.lightmap_offset + size * idx:
                              self.lightmap_offset + size * (idx + 1)]
        )).reshape(shape)

        return lightmap


@dataclasses.dataclass
class TexInfo:
    bsp: 'Q2Bsp'
    vec_s: float
    dist_s: float
    vec_t: float
    dist_t: float
    flags: TexInfoFlags
    value: int
    texture: str
    next_: int


    def vert_to_tex_coords(self, vert):
        return [np.dot(vert, self.vec_s) + self.dist_s, np.dot(vert, self.vec_t) + self.dist_t]


class MalformedBspFile(Exception):
    pass


def _read(f, n):
    b = f.read(n)
    if len(b) < n:
        raise MalformedBspFile("File ended unexpectedly")
    return b


def _read_struct(f, struct_fmt):
    size = struct.calcsize(struct_fmt)
    out = struct.unpack(struct_fmt, self._read(f, size))
    return out


def _read_lump(f, dir_entry, struct_fmt, post_func=None):
    size = struct.calcsize(struct_fmt)
    f.seek(dir_entry.offset)
    if dir_entry.size % size != 0:
        raise MalformedBspFile("Invalid lump size")
    out = [struct.unpack(struct_fmt, _read(f, size)) for _ in range(0, dir_entry.size, size)]
    if post_func:
        out = [post_func(*x) for x in out]
    return out


def _read_dir_entry(f):
    fmt = "<II"
    size = struct.calcsize(fmt)
    return _DirEntry(*struct.unpack(fmt, _read(f, size)))


def _read_header(f):
    magic = f.read(4)
    if magic != b'IBSP':
        raise MalformedBspException(f'Bad magic {magic}')

    version, = struct.unpack('<I', f.read(4))
    if version != 38:
        raise MalformedBspFile(f'Invalid version {version}')


def _read_dir_entries(f) -> list[_DirEntry]:
    return [_read_dir_entry(f) for _ in range(19)]


class Q2Bsp:
    vertices: list[tuple[float, float, float]]
    faces: list[Face]
    edges: list[tuple[int, int]]
    edge_lists: list[int]

    def __init__(self, f):
        _read_header(f)
        dir_entries = _read_dir_entries(f)
        self._dir_entries = dir_entries

        def read_face(plane_id, side, edge_list_idx, num_edges, texinfo_id, s1, s2, s3, s4,
                      lightmap_offset):
            return Face(self, plane_id, bool(side), edge_list_idx, num_edges,
                        texinfo_id, [s1, s2, s3, s4], lightmap_offset)
        self.vertices = _read_lump(f, dir_entries[2], "<fff")
        self.faces = _read_lump(f, dir_entries[6], "<HHLHHBBBBl", read_face)
        self.edges = _read_lump(f, dir_entries[11], "<HH")
        self.edge_list = _read_lump(f, dir_entries[12], "<l", lambda x: x)
        def read_texinfo(vs1, vs2, vs3, ds, vt1, vt2, vt3, dt, flags, value, texture_bytes, next_):
            texture_str = texture_bytes[:texture_bytes.index(b'\0')].decode('ascii')
            return TexInfo(self, (vs1, vs2, vs3), ds, (vt1, vt2, vt3), dt,
                           TexInfoFlags(flags), value, texture_str, next_)
        self.texinfo = _read_lump(f, dir_entries[5], "<ffffffffLL32sL", read_texinfo)

        def read_plane(n1, n2, n3, d, plane_type):
            return Plane((n1, n2, n3), d, PlaneType(plane_type))
        self.planes = _read_lump(f, dir_entries[1], "<ffffl", read_plane)

        lightmap_dir_entry = dir_entries[7]
        f.seek(lightmap_dir_entry.offset)
        self.lightmap = _read(f, lightmap_dir_entry.size)

        entities_dir_entry = dir_entries[0]
        f.seek(entities_dir_entry.offset)
        entities_str = _read(f, entities_dir_entry.size)
        entities_str = entities_str[:entities_str.index(b'\0')].decode('ascii')
        self.entities = ent.parse_entities(entities_str)


def rewrite_lightmap(in_f, new_lms: dict[Face, np.ndarray], out_f):
    _read_header(in_f)
    lightmap_dir_entry = _read_dir_entries(in_f)[7]
    in_f.seek(0)
    bsp_array = np.frombuffer(in_f.read(), dtype=np.uint8).copy()

    for face, lm_array in new_lms.items():
        assert face.lightmap_shape + (3,) == lm_array.shape
        assert lm_array.dtype == np.uint8
        offs = face.lightmap_offset + lightmap_dir_entry.offset
        h, w = face.lightmap_shape
        assert face.lightmap_offset + w * h * 3 <= lightmap_dir_entry.size
        bsp_array[offs:offs + w * h * 3] = lm_array.ravel()

    out_f.write(bsp_array.tobytes())
