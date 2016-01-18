set (vfgpu_sources
  puncture.cu
)

cuda_add_library (vfgpu STATIC ${vfgpu_sources})

if (WITH_NETCDF) # dump intermediate data
  target_link_libraries (vfgpu ${CUDA_curand_LIBRARY} ${NETCDF_LIBRARIES})
else ()
  target_link_libraries (vfgpu ${CUDA_curand_LIBRARY})
endif ()
