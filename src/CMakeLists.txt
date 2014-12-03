set (viewer_sources
  trackball.cpp
  widget.cpp
  viewer.cpp
)

add_executable (viewer ${viewer_sources})
target_link_libraries (viewer
  ${QT_LIBRARIES}
  ${OPENGL_LIBRARIES}
)

set (extractor_sources
  extractor.cpp
)

add_executable (main main.cpp ${extractor_sources})
target_link_libraries (main 
  ${LIBMESH_LIBRARY}
)
