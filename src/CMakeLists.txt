include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

configure_file (def.h.cmake def.h)

subdirs (vortex io extractor)

if (WITH_QT)
  subdirs (viewer)
endif ()
