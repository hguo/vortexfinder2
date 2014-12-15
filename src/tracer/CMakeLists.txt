set (extractor_sources
  Tracer.cpp
  Condor2Tracer.cpp
  GLGPUTracer.cpp
)

if (WITH_LIBMESH)
  set (extractor_sources ${extractor_sources}
    Condor2Tracer.cpp)
endif ()

add_library (gltracer ${extractor_sources})
link_libraries (gltracer glcommon glio)

add_executable (tracer main.cpp)
target_link_libraries (tracer gltracer)
