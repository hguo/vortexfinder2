set (extractor_sources
  Extractor.cpp
  Condor2Extractor.cpp
  GLGPUExtractor.cpp
)

add_library (glextractor ${extractor_sources})
link_libraries (glextractor glcommon glio ${LIBMESH_LIBRARY})

add_executable (extractor main.cpp)
target_link_libraries (extractor glextractor)
