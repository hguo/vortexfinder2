set (extractor_sources
  extractor.cpp
)

add_library (glextractor ${extractor_sources})
link_libraries (glextractor ${LIBMESH_LIBRARY})

add_executable (extractor main.cpp)
target_link_libraries (extractor glextractor)
