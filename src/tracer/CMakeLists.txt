set (extractor_sources
  Tracer.cpp
)

add_library (gltracer ${extractor_sources})
link_libraries (gltracer glcommon glio)

add_executable (tracer_glgpu ex_glgpu.cpp)
target_link_libraries (tracer_glgpu gltracer)

add_executable (tracer_condor2 ex_condor2.cpp)
target_link_libraries (tracer_condor2 gltracer)
