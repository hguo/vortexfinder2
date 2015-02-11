set (extractor_sources
  Puncture.cpp
  Extractor.cpp
  # GLGPUExtractor.cpp
)

if (WITH_LIBMESH)
  set (extractor_sources ${extractor_sources}
    Condor2Extractor.cpp)
endif ()

add_library (glextractor ${extractor_sources})
link_libraries (glextractor glcommon glio)

if (WITH_LIBMESH)
  add_executable (extractor_condor2 ex_condor2.cpp)
  target_link_libraries (extractor_condor2 glextractor)
endif ()

add_executable (extractor_presave ex_presave.cpp)
target_link_libraries (extractor_presave glextractor)

# add_executable (extractor_glgpu ex_glgpu.cpp)
# target_link_libraries (extractor_glgpu glextractor)
