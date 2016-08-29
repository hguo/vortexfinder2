set (volren_sources
  rc.cu)

if (WITH_CUDA)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  cuda_add_library (glvolren STATIC ${volren_sources})
endif ()
