set (vfgpu_sources
  puncture.cu
)

cuda_add_library (vfgpu STATIC ${vfgpu_sources})
target_link_libraries (vfgpu ${CUDA_curand_LIBRARY})
# target_link_libraries (vfgpu ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
