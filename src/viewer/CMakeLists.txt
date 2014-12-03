set (viewer_sources
  trackball.cpp
  widget.cpp
  main.cpp
)

add_executable (viewer ${viewer_sources})
target_link_libraries (viewer
  ${QT_LIBRARIES}
  ${OPENGL_LIBRARIES}
)
