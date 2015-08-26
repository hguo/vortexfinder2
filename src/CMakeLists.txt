include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

configure_file (def.h.cmake def.h)

add_subdirectory (common)
add_subdirectory (io)
add_subdirectory (extractor)
add_subdirectory (analysis)

if (WITH_PARAVIEW)
  add_subdirectory (paraview_plugins)
endif ()

if (WITH_QT)
  add_subdirectory (viewer)
endif ()

if (WITH_TESTS)
  add_subdirectory (tests)
endif ()
