set (extractor_sources
  Tracer.cpp
  Condor2Tracer.cpp
  GLGPUTracer.cpp
)

add_library (gltracer ${extractor_sources})
link_libraries (gltracer glvortex ${LIBMESH_LIBRARY})

add_executable (tracer main.cpp)
target_link_libraries (tracer gltracer)
