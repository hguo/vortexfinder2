if (WITH_CUDA)
  set (vf2gpu_sources
    vf2gpu.cu
    vf2gpu.cpp
  )
  cuda_add_library (vf2gpu ${vf2gpu_sources})
endif ()
