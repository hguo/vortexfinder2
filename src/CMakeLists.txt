set (viewer_sources
  trackball.cpp
  widget.cpp
  viewer.cpp
)

add_executable (viewer ${viewer_sources})
target_link_libraries (viewer
  ${QT_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARY}
  ${GLUT_LIBRARY}
)

add_executable (main main.cpp)
target_link_libraries (main 
  ${LIBMESH_LIBRARY}
)

add_executable (main2 main2.cpp)
target_link_libraries (main2 
  ${LIBMESH_LIBRARY}
)
