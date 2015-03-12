set (CMAKE_CXX_FLAGS "")

set (volren_sources
  rc.cu)

if (WITH_CUDA)
  cuda_add_library (gl_volren ${volren_sources})
endif ()
