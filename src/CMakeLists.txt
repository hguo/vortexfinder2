include_directories (${CMAKE_CURRENT_SOURCE_DIR})

subdirs (vortex extractor)

if (WITH_QT)
  subdirs (viewer)
endif ()
