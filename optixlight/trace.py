import ctypes
import logging

import cupy as cp
import numpy as np
import optix


logger = logging.getLogger(__name__)


def _create_ctx() -> optix.DeviceContext:
    def log(level, tag, msg):
        logger.info(f"[{level:>2}][{tag:>12}]: {msg}")

    ctx_options = optix.DeviceContextOptions(
        logCallbackFunction = log,
        logCallbackLevel    = 4
    )
    if optix.version()[1] >= 2:
        ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL
    cu_ctx = 0
    return optix.deviceContextCreate( cu_ctx, ctx_options )


def _array_to_device_memory( numpy_array, stream=cp.cuda.Stream() ):
    byte_size = numpy_array.size * numpy_array.dtype.itemsize
    h_ptr = ctypes.c_void_p(numpy_array.ctypes.data)
    d_mem = cp.cuda.memory.alloc(byte_size)
    d_mem.copy_from_async(h_ptr, byte_size, stream)

    return d_mem


def _create_accel(ctx: optix, tris: np.ndarray, tex_vecs: np.ndarray) \
        -> optix.TraversableHandle:
    d_vertices = _array_to_device_memory(tris)

    accel_options = optix.AccelBuildOptions(
        buildFlags = int( optix.BUILD_FLAG_ALLOW_COMPACTION ),
        operation = optix.BUILD_OPERATION_BUILD
    )

    triangle_input = optix.BuildInputTriangleArray()
    triangle_input.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
    triangle_input.vertexStrideInBytes = tris.dtype.itemsize * 3
    triangle_input.numVertices = len(tris) * 3
    triangle_input.vertexBuffers = [d_vertices.ptr]
    triangle_input.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT]
    triangle_input.numSbtRecords = 1

    gas_buffer_sizes = ctx.accelComputeMemoryUsage([accel_options], [triangle_input])
    d_temp_buffer_gas = cp.cuda.alloc(gas_buffer_sizes.tempSizeInBytes)
    d_gas_output_buffer = cp.cuda.alloc(gas_buffer_sizes.outputSizeInBytes)
    d_result = cp.array([0], dtype='u8')

    emit_property = optix.AccelEmitDesc(
        type = optix.PROPERTY_TYPE_COMPACTED_SIZE,
        result = d_result.data.ptr
    )

    gas_handle = ctx.accelBuild(
        0,    # CUDA stream
        [ accel_options ],
        [ triangle_input ],
        d_temp_buffer_gas.ptr,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer.ptr,
        gas_buffer_sizes.outputSizeInBytes,
        [emit_property]
    )

    compacted_gas_size = cp.asnumpy(d_result)[0]

    if compacted_gas_size < gas_buffer_sizes.outputSizeInBytes:
        # TODO: reallocate with call to `accelCompact`
        pass

    return gas_handle


def trace(tris: np.ndarray,
          light_origin: np.ndarray,
          tex_vecs: np.ndarray,
          output_shape: tuple[int, int]) -> np.ndarray:
    """
    Arguments:
        tris: (n, 3, 3) float array of triangle vertices.
        light_origin (3,) float array with the light origin.
        tex_vecs: (n, 2, 4) float array of texture coordinates.
            `v[i, 0] @ (x, y, z, 1)` gives the s texture coordinate in the
            output image, and `v[i, 1] @ (x, y, z, 1)` gives the t texture
            coordinate.
        output_shape: shape of the output image.

    Returns:
        The output image.
    """

    ctx = _create_ctx()
    gas_handle = _create_accel(ctx, tris, tex_vecs)
