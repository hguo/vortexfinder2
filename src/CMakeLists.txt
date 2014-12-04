include_directories (${CMAKE_CURRENT_SOURCE_DIR})

subdirs (vortex io extractor)

if (WITH_QT)
  subdirs (viewer)
endif ()
