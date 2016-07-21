set (extractor_sources
  Tracer.cpp
)

add_library (gltracer ${extractor_sources})
link_libraries (gltracer glcommon glio)
