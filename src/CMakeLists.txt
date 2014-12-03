include_directories (${CMAKE_CURRENT_SOURCE_DIR})

subdirs (extractor)

if (WITH_QT)
  subdirs (viewer)
endif ()
