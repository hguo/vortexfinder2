set (extractor_sources
  Extractor.cpp
  StochasticExtractor.cpp
)
  
add_library (glextractor STATIC ${extractor_sources})

if (WITH_CUDA)
  add_subdirectory (vfgpu)
  target_link_libraries (glextractor PUBLIC glcommon glio vfgpu)
else ()
  target_link_libraries (glextractor PUBLIC glcommon glio)
endif ()

#install (TARGETS glextractor DESTINATION ${VTK_INSTALL_LIBRARY_DIR})
