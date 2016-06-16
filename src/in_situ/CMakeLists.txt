if (WITH_CUDA)
  set (vf2gpu_sources
    vf2gpu.cpp
  )
  add_library (vf2gpu ${vf2gpu_sources})
endif ()
