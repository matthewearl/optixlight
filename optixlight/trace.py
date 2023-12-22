import ctypes
import logging
import pathlib
import subprocess

import cupy as cp
import numpy as np
import optix as ox
import optix.struct


logger = logging.getLogger(__name__)


def _make_aligned_dtype(fields, align):
    names = [name for fmt, name in fields]
    formats = [fmt for fmt, name in fields]
    itemsize = optix.struct._aligned_itemsize(formats, align)
    return np.dtype(
        {
            'names'     : names,
            'formats'   : formats,
            'itemsize'  : itemsize,
        },
        align=True,
    )


entry_dtype = _make_aligned_dtype([
    ('f4', 'p'),
    ('u4', 'face_idx'),
    ('3u1', 'color'),
    ('u1', 'pad'),  # align=True does not match CUDA alignment perfectly
    ('2u1', 'tc'),
], 8)


def _create_ctx() -> ox.DeviceContext:
    def log(level, tag, msg):
        logger.info(f"[{level:>2}][{tag:>12}]: {msg}")
    ctx = ox.DeviceContext(validation_mode=True,
                           log_callback_function=log,
                           log_callback_level=3)
    return ctx


def _create_accel(ctx: ox.DeviceContext, tris: np.ndarray,
                  num_faces: int,
                  face_idxs: np.ndarray) -> ox.AccelerationStructure:
    build_input = ox.BuildInputTriangleArray(
        tris.astype(np.float32).reshape(-1, 3),
        num_sbt_records=num_faces,
        sbt_record_offset_buffer=face_idxs.astype(np.uint32),
        flags=[ox.GeometryFlags.NONE] * num_faces
    )
    gas = ox.AccelerationStructure(ctx, build_input, compact=True)
    return gas


def _create_pipeline_options() -> ox.PipelineCompileOptions:
    return ox.PipelineCompileOptions(
        traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
        num_payload_values=1,
        num_attribute_values=3,
        exception_flags=ox.ExceptionFlags.NONE,
        pipeline_launch_params_variable_name="params"
    )


def _source_to_ptx():
    src_dir = pathlib.Path(__file__).parent
    try:
        result = subprocess.run(
            [
                'nvcc',
                '-ptx', '--use_fast_math', '-lineinfo', '-std=c++11',
                '-I/home/matt/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/SDK',
                '-I/home/matt/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/include',
                'optixlight.cu'
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=src_dir
        )
        for line in result.stdout.strip().split('\n'):
            logger.error(f'nvcc stdout: {line}')
    except subprocess.CalledProcessError as e:
        for line in e.stderr.strip().split('\n'):
            logger.error(f'nvcc stderr: {line}')
        raise Exception('Failed to compile')

    with (src_dir / 'optixlight.ptx').open('rb') as f:
        ptx_data = f.read()

    return ptx_data


def _create_module(ctx: ox.DeviceContext,
                   pipeline_options: ox.PipelineCompileOptions) -> ox.Module:

    # For some reason compiling with `ox.Module` while importing curand_kernel.h
    # doesn't work for me.  Instead let's invoke `nvcc` ourselves and pass the
    # output ptx file to `ox.Module`.
    ptx_data = _source_to_ptx()
    return ox.Module(
        ctx, ptx_data, pipeline_compile_options=pipeline_options,
    )


def _create_program_groups(ctx: ox.DeviceContext, module: ox.Module) \
        -> list[ox.ProgramGroup]:
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_CH="__closesthit__ch")
    return raygen_grp, miss_grp, hit_grp


def _create_pipeline(ctx: ox.DeviceContext,
                     program_groups: list[ox.ProgramGroup],
                     pipeline_compile_options: ox.PipelineCompileOptions) \
                         -> ox.Pipeline:
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1,
                                       debug_level=ox.CompileDebugLevel.FULL)
    pipeline = ox.Pipeline(ctx,
                           compile_options=pipeline_compile_options,
                           link_options=link_opts,
                           program_groups=program_groups)
    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 1)  # max_dc_depth

    return pipeline


def _create_sbt(prog_groups: list[ox.ProgramGroup],
                num_faces: int) -> ox.ShaderBindingTable:
    raygen_grp, miss_grp, hit_grp = prog_groups

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp)

    hit_field_names = ('idx',)
    hit_field_formats = ('u4')
    hit_records = []
    for i in range(num_faces):
        hit_record = ox.SbtRecord(hit_grp,
                                  names=hit_field_names,
                                  formats=hit_field_formats)
        hit_record['idx'] = i
        hit_records.append(hit_record)
    hit_sbt = np.concatenate([hr.array for hr in hit_records])
    hit_sbt = np.array(hit_sbt, dtype=hit_records[0].array.dtype)

    return ox.ShaderBindingTable(raygen_record=raygen_sbt,
                                 miss_records=miss_sbt,
                                 hitgroup_records=hit_sbt)

def _find_tangents(normal):
    """Find two vectors orthogonal to the given vector"""
    temp = np.zeros(3)
    temp[np.argmin(normal)] = 1.0
    tangent1 = np.cross(normal, temp)
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)

    return tangent1, tangent2


def _launch(pipeline: ox.Pipeline, sbt: ox.ShaderBindingTable,
            gas: ox.AccelerationStructure,
            num_threads: int,
            rays_per_thread: int,
            light_origin: np.ndarray,
            source_entries: np.ndarray,
            source_cdf: np.ndarray,
            normals: np.ndarray,
            world_to_tcs: np.ndarray,
            tc_to_worlds: np.ndarray,
            lm_shapes: np.ndarray,
            lm_offsets: np.ndarray) -> np.ndarray:
    h_counts = np.zeros(len(lm_shapes) + 1, dtype='u4')
    d_counts = cp.array(h_counts)

    output_shape = max(3 * shape[0] * shape[1] + offset
                       for shape, offset
                       in zip(lm_shapes, lm_offsets, strict=True))
    h_output = np.zeros(output_shape, dtype='f4')
    d_output = cp.array(h_output)

    d_source_entries = optix.struct.array_to_device_memory(source_entries)
    d_source_cdf = cp.array(source_cdf)

    # Make face info.  Structs seem to have a size that is a multiple of
    # SBT_RECORD_ALIGNMENT, but I don't know how robust this is...
    dtype = _make_aligned_dtype([
        ('4f4', 'world_to_tc_0'),
        ('4f4', 'world_to_tc_1'),
        ('3f4', 'tc_to_world_0'),
        ('3f4', 'tc_to_world_1'),
        ('3f4', 'tc_to_world_2'),
        ('3f4', 'normal'),
        ('3f4', 'tangent1'),
        ('3f4', 'tangent2'),
        ('u4', 'lm_width'),
        ('u4', 'lm_height'),
        ('u4', 'lm_offset'),
    ], 16)
    h_face_info = np.array([
        (world_to_tc[0], world_to_tc[1],
         tc_to_world[0], tc_to_world[1], tc_to_world[2],
         normal, *_find_tangents(normal),
         lm_shape[1], lm_shape[0],
         lm_offset)
        for normal, world_to_tc, tc_to_world, lm_shape, lm_offset
        in zip(normals, world_to_tcs, tc_to_worlds,
               lm_shapes, lm_offsets, strict=True)
    ], dtype=dtype)
    d_face_info = optix.struct.array_to_device_memory(h_face_info)

    params_tmp = [
        ('u8', 'trav_handle'),
        ('u8', 'counts'),
        ('u8', 'output'),
        ('u8', 'faces'),
        ('u8', 'source_entries'),
        ('u8', 'source_cdf'),
        ('u4', 'num_source_entries'),
        ('u4', 'seed'),
        ('u4', 'rays_per_thread'),
        ('f4', 'lx'),
        ('f4', 'ly'),
        ('f4', 'lz'),
    ]
    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                   formats=[p[0] for p in params_tmp])
    params['trav_handle'] = gas.handle
    params['counts'] = d_counts.data.ptr
    params['output'] = d_output.data.ptr
    params['faces'] = d_face_info.ptr
    params['source_entries'] = d_source_entries.ptr
    params['source_cdf'] = d_source_cdf.data.ptr
    params['num_source_entries'] = len(source_entries)
    params['seed'] = 0
    params['rays_per_thread'] = rays_per_thread
    params['lx'] = light_origin[0]
    params['ly'] = light_origin[1]
    params['lz'] = light_origin[2]

    stream = cp.cuda.Stream()
    pipeline.launch(sbt, dimensions=(num_threads, 1, 1), params=params,
                    stream=stream)
    stream.synchronize()

    return cp.asnumpy(d_counts), cp.asnumpy(d_output) / num_threads


def trace(tris: np.ndarray,
          light_origin: np.ndarray,
          source_entries: np.ndarray,
          source_cdf: np.ndarray,
          face_idxs: np.ndarray,
          normals: np.ndarray,
          world_to_tcs: np.ndarray,
          tc_to_worlds: np.ndarray,
          lm_shapes: np.ndarray,
          lm_offsets: np.ndarray) -> np.ndarray:
    """
    Arguments:
        tris: (n, 3, 3) float array of triangle vertices.
        light_origin: (3,) float array with the light origin.
        source_entries: (n_sources,) light source records.
        source_cdf: (n_sources,) uint32 array light source cumulative
            distribution function, scaled by (1<<32).
        face_idxs: (n,) int array of face indices, one per tri.
        normals: (m, 3) float array of face normals.
        world_to_tcs: (m, 2, 4) float array of texture coordinate maps.
            `v[i, 0] @ (x, y, z, 1)` gives the s texture coordinate of the
            `i`'th face in the output image, and `v[i, 1] @ (x, y, z, 1)` gives
            the t texture coordinate.
        tc_to_worlds: (m, 3, 3) float array of inverse texture coordinate maps.
            Each element maps augmented texture coordinates [s, t, 1] into world
            space coordinates [x, y, z], for the corresponding face.
        lm_shapes: (m, 2) int shape of each face's lightmap.
        lm_offsets: (m,) int array of each face's offset.

    Returns:
        The output image.
    """

    ctx = _create_ctx()
    gas_handle = _create_accel(ctx, tris, len(world_to_tcs), face_idxs)
    pipeline_options = _create_pipeline_options()
    module = _create_module(ctx, pipeline_options)
    prog_groups = _create_program_groups(ctx, module)
    pipeline = _create_pipeline(ctx, prog_groups, pipeline_options)
    sbt = _create_sbt(prog_groups, len(world_to_tcs))

    num_threads = 10_000
    counts, output = _launch(pipeline, sbt, gas_handle,
                             num_threads,
                             1_000_000 // num_threads,
                             light_origin,
                             source_entries,
                             source_cdf,
                             normals, world_to_tcs, tc_to_worlds,
                             lm_shapes, lm_offsets)

    return output, counts
