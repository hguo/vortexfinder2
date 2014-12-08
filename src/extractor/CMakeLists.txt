set (extractor_sources
  Extractor.cpp
  Condor2Extractor.cpp
)

add_library (glextractor ${extractor_sources})
link_libraries (glextractor glvortex ${LIBMESH_LIBRARY})

add_executable (extractor main.cpp)
target_link_libraries (extractor glextractor)
