set (extractor_sources
  Extractor.cpp
)

set (extractor_sources_cu
  Extractor.cu
)

if (WITH_CUDA)
  cuda_add_library (glextractor STATIC ${extractor_sources} ${extractor_sources_cu})
  target_link_libraries (glextractor glcommon glio ${CUDA_LIBRARIES})
else ()
  add_library (glextractor STATIC ${extractor_sources})
  target_link_libraries (glextractor PUBLIC glcommon glio)
endif ()

#install (TARGETS glextractor DESTINATION ${VTK_INSTALL_LIBRARY_DIR})
