set (extractor_sources
  Extractor.cpp
  Condor2Extractor.cpp
  GLGPUExtractor.cpp
)

add_library (glextractor ${extractor_sources})
link_libraries (glextractor glcommon glio ${LIBMESH_LIBRARY})

add_executable (extractor_condor2 ex_condor2.cpp)
target_link_libraries (extractor_condor2 glextractor)

add_executable (extractor_glgpu ex_glgpu.cpp)
target_link_libraries (extractor_glgpu glextractor)
