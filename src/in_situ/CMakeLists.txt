if (WITH_CUDA)
  set (vf2gpu_sources
    vf2gpu.cu
    vf2gpu.cpp
  )
  cuda_add_library (vf2gpu ${vf2gpu_sources})
endif ()

install (TARGETS vf2gpu
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ GROUP_WRITE WORLD_READ)

install (FILES vf2gpu.h
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ GROUP_WRITE WORLD_READ)
