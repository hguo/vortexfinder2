set (CMAKE_CXX_FLAGS "")

set (volren_sources
  rc.cu)

if (WITH_CUDA)
  cuda_add_library (glvolren ${volren_sources})
endif ()
