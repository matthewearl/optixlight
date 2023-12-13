import ctypes
import logging
import os

import cupy as cp
import numpy as np
import optix
from pynvrtc.compiler import Program


logger = logging.getLogger(__name__)

include_paths = [
    "/home/matt/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64/include",
    "/home/matt/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64/SDK",
]
cuda_tk_path = "/usr/include"
stddef_path = "/usr/include/linux"


def _round_up( val, mult_of ):
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def _get_aligned_itemsize(formats, alignment):
    names = []
    for i in range(len(formats)):
        names.append('x'+str(i))

    temp_dtype = np.dtype({
        'names'   : names,
        'formats' : formats,
        'align'   : True
    })
    return _round_up(temp_dtype.itemsize, alignment)


def _compile_cuda(cuda_file):
    with open(cuda_file, 'rb') as f:
        src = f.read()

    nvrtc_dll = os.environ.get('NVRTC_DLL')
    if nvrtc_dll is None:
        nvrtc_dll = ''
    logger.info("NVRTC_DLL = %s", nvrtc_dll)
    prog = Program( src.decode(), cuda_file,
                    lib_name= nvrtc_dll )
    compile_options = [
        '-use_fast_math',
        '-lineinfo',
        '-default-device',
        '-std=c++11',
        '-rdc',
        'true',
        f'-I{cuda_tk_path}',
    ] + [f'-I{include_path}' for include_path in include_paths]

    if (optix.version()[1] == 0):
        compile_options.append( f'-I{stddef_path}' )

    ptx = prog.compile( compile_options )
    return ptx


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


def _create_accel(ctx: optix.DeviceContext, tris: np.ndarray,
                  tex_vecs: np.ndarray) -> optix.TraversableHandle:
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


def _create_pipeline_options() -> optix.PipelineCompileOptions:
    return optix.PipelineCompileOptions(
        usesMotionBlur = False,
        traversableGraphFlags = int( optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS ),
        numPayloadValues = 1,
        numAttributeValues = 1,  # TODO: Update
        exceptionFlags = int(optix.EXCEPTION_FLAG_DEBUG),
        pipelineLaunchParamsVariableName = "params",
        usesPrimitiveTypeFlags = optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE
    )


def _create_module(ctx: optix.DeviceContext,
                   pipeline_options: optix.PipelineCompileOptions,
                   ptx) -> optix.Module:
    logger.info("Creating OptiX module ...")
    module_options = optix.ModuleCompileOptions(
        maxRegisterCount = optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel       = optix.COMPILE_DEBUG_LEVEL_DEFAULT
        )

    module, log = ctx.moduleCreateFromPTX(
        module_options,
        pipeline_options,
        ptx
        )
    logger.info("\tModule create log: <<<{}>>>".format(log))
    return module


def _create_program_groups(ctx: optix.DeviceContext, module: optix.Module) \
        -> list[optix.ProgramGroup]:
    logger.info( "Creating program groups ... " )

    raygen_prog_group_desc = optix.ProgramGroupDesc()
    raygen_prog_group_desc.raygenModule = module
    raygen_prog_group_desc.raygenEntryFunctionName = "__raygen__rg"
    (raygen_prog_group,), log = ctx.programGroupCreate([raygen_prog_group_desc])
    logger.info("\tProgramGroup raygen create log: <<<{}>>>".format(log))

    miss_prog_group_desc = optix.ProgramGroupDesc()
    miss_prog_group_desc.missModule = module
    miss_prog_group_desc.missEntryFunctionName = "__miss__ms"
    (miss_prog_group,), log = ctx.programGroupCreate([miss_prog_group_desc])
    logger.info("\tProgramGroup miss create log: <<<{}>>>".format(log))

    hitgroup_prog_group_desc = optix.ProgramGroupDesc()
    hitgroup_prog_group_desc.hitgroupModuleCH = module
    hitgroup_prog_group_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"
    (hitgroup_prog_group,), log = ctx.programGroupCreate(
        [hitgroup_prog_group_desc]
    )
    logger.info("\tProgramGroup hitgroup create log: <<<{}>>>".format(log))

    return [raygen_prog_group, miss_prog_group, hitgroup_prog_group]


def _create_pipeline(ctx: optix.DeviceContext,
                     program_groups: list[optix.ProgramGroup],
                     pipeline_compile_options: optix.PipelineCompileOptions) \
                         -> optix.Pipeline:
    logger.info("Creating pipeline ... ")

    max_trace_depth = 1
    pipeline_link_options = optix.PipelineLinkOptions()
    pipeline_link_options.maxTraceDepth = max_trace_depth
    pipeline_link_options.debugLevel = optix.COMPILE_DEBUG_LEVEL_FULL

    log = ""
    pipeline = ctx.pipelineCreate(
        pipeline_compile_options,
        pipeline_link_options,
        program_groups,
        log
    )

    stack_sizes = optix.StackSizes()
    for prog_group in program_groups:
        optix.util.accumulateStackSizes(prog_group, stack_sizes)

    (dc_stack_size_from_trav, dc_stack_size_from_state, cc_stack_size) = \
        optix.util.computeStackSizes(
            stack_sizes,
            max_trace_depth,
            0,  # maxCCDepth
            0   # maxDCDepth
            )

    pipeline.setStackSize(
        dc_stack_size_from_trav,
        dc_stack_size_from_state,
        cc_stack_size,
        1   # maxTraversableDepth
    )

    return pipeline


def _create_sbt(prog_groups: list[optix.ProgramGroup]) \
        -> optix.ShaderBindingTable:
    logging.debug("Creating sbt ... ")

    (raygen_prog_group, miss_prog_group, hitgroup_prog_group) = prog_groups

    header_format = '{}B'.format( optix.SBT_RECORD_HEADER_SIZE )

    #
    # raygen record
    #
    formats = [header_format]
    itemsize = _get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
    dtype = np.dtype( {
        'names'     : ['header'],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'align'     : True
        })
    h_raygen_sbt = np.array([(0,)], dtype=dtype)
    optix.sbtRecordPackHeader(raygen_prog_group, h_raygen_sbt)
    d_raygen_sbt = _array_to_device_memory(h_raygen_sbt)

    #
    # miss record
    #
    formats = [header_format]
    itemsize = _get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
    dtype = np.dtype( {
        'names'     : ['header'],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'align'     : True
        })
    h_miss_sbt = np.array([(0,)], dtype=dtype)
    optix.sbtRecordPackHeader(miss_prog_group, h_miss_sbt)
    d_miss_sbt = _array_to_device_memory(h_miss_sbt)

    #
    # hitgroup record
    #
    formats = [header_format]
    itemsize = _get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
    dtype = np.dtype( {
        'names'     : ['header'],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'align'     : True
        } )
    h_hitgroup_sbt = np.array([(0,)], dtype=dtype)
    optix.sbtRecordPackHeader(hitgroup_prog_group, h_hitgroup_sbt)
    d_hitgroup_sbt = _array_to_device_memory( h_hitgroup_sbt )

    return optix.ShaderBindingTable(
        raygenRecord=d_raygen_sbt.ptr,
        missRecordBase=d_miss_sbt.ptr,
        missRecordStrideInBytes=h_miss_sbt.dtype.itemsize,
        missRecordCount=1,
        hitgroupRecordBase=d_hitgroup_sbt.ptr,
        hitgroupRecordStrideInBytes=h_hitgroup_sbt.dtype.itemsize,
        hitgroupRecordCount=1
    )


def _launch(pipeline: optix.Pipeline, sbt: optix.ShaderBindingTable,
            trav_handle: optix.TraversableHandle,
            num_tris: int,
            width: int,
            height: int,
            light_origin: np.ndarray) -> np.ndarray:
    logger.info( "Launching ... " )

    h_counts = np.zeros(num_tris + 1, dtype='u4')
    d_counts = cp.array(h_counts)

    params = [
        ('u4', 'seed', 0),
        ('u8', 'counts', d_counts.data.ptr),
        ('u4', 'width', width),
        ('f4', 'light_origin_x', light_origin[0]),
        ('f4', 'light_origin_y', light_origin[1]),
        ('f4', 'light_origin_z', light_origin[2]),

        ('u8', 'trav_handle',  trav_handle)
    ]

    formats = [x[0] for x in params]
    names = [x[1] for x in params]
    values = [x[2] for x in params]
    itemsize = _get_aligned_itemsize(formats, 8)
    params_dtype = np.dtype({
        'names'   : names,
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True, 
    })
    h_params = np.array([tuple(values)], dtype=params_dtype)
    d_params = _array_to_device_memory(h_params)

    stream = cp.cuda.Stream()
    optix.launch(
        pipeline,
        stream.ptr,
        d_params.ptr,
        h_params.dtype.itemsize,
        sbt,
        width,
        height,
        1 # depth
    )

    stream.synchronize()

    return cp.asnumpy(d_counts)


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
    optixlight_cu = os.path.join(os.path.dirname(__file__), 'optixlight.cu')
    optixlight_ptx = _compile_cuda(optixlight_cu)

    ctx = _create_ctx()
    gas_handle = _create_accel(ctx, tris, tex_vecs)
    pipeline_options = _create_pipeline_options()
    module = _create_module(ctx, pipeline_options, optixlight_ptx)
    prog_groups = _create_program_groups(ctx, module)
    pipeline = _create_pipeline(ctx, prog_groups, pipeline_options)
    sbt = _create_sbt(prog_groups)

    counts = _launch(pipeline, sbt, gas_handle, len(tris), 100, 100,
                     light_origin)
    print(counts)
