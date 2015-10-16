set (extractor_sources
  Extractor.cpp
)

add_library (glextractor STATIC ${extractor_sources})
target_link_libraries (glextractor PUBLIC glcommon glio ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
#install (TARGETS glextractor DESTINATION ${VTK_INSTALL_LIBRARY_DIR})
