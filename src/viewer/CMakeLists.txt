set (viewer_sources
  trackball.cpp
  widget.cpp
  main.cpp
)

if (APPLE)
  add_definitions ("-Wno-deprecated")
endif ()

add_executable (viewer ${viewer_sources})
target_link_libraries (viewer 
  glcommon
  ${QT_LIBRARIES}
  ${OPENGL_LIBRARIES}
)
set_target_properties (viewer PROPERTIES AUTOMOC ON)
