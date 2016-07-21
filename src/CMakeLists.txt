include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

configure_file (def.h.cmake def.h)

add_subdirectory (common)
add_subdirectory (io)
add_subdirectory (extractor)
add_subdirectory (tracer)

if (WITH_PARAVIEW)
  add_subdirectory (paraview_plugins)
else ()
  add_subdirectory (analysis)
endif ()

if (WITH_QT)
  add_subdirectory (viewer)
endif ()

#if (WITH_IN_SITU)
#  add_subdirectory (in_situ)
#endif ()

if (WITH_TESTS)
  add_subdirectory (tests)
endif ()
