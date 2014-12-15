set (viewer_sources
  trackball.cpp
  widget.cpp
  mainWindow.cpp
  main.cpp
)

qt4_wrap_ui (viewer_headers 
  mainWindow.ui)

if (APPLE) # remove anoying OpenGL deprecated warnings
  add_definitions ("-Wno-deprecated")
endif ()

add_executable (viewer ${viewer_sources} ${viewer_headers})
target_link_libraries (viewer 
  glcommon
  ${QT_LIBRARIES}
  ${OPENGL_LIBRARIES}
)
set_target_properties (viewer PROPERTIES AUTOMOC ON)
