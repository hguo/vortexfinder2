set (viewer_sources
  trackball.cpp
  widget.cpp
  storyLineWidget.cpp
  mainWindow.cpp
)

qt4_wrap_ui (viewer_headers 
  mainWindow.ui)

if (APPLE) # remove anoying OpenGL deprecated warnings
  add_definitions ("-Wno-deprecated")
endif ()

add_library (glviewer ${viewer_sources} ${viewer_headers})
target_link_libraries (glviewer 
  glcommon
  glio
  ${QT_LIBRARIES}
  ${GLUT_LIBRARY}
  ${OPENGL_LIBRARIES}
)
if (WITH_VTK)
  target_link_libraries (glviewer ${VTK_LIBRARIES})
endif ()
set_target_properties (glviewer PROPERTIES AUTOMOC ON)

add_executable (viewer main.cpp)
target_link_libraries (viewer glviewer)

add_executable (viewer1 main1.cpp)
target_link_libraries (viewer1 glviewer)

add_executable (storyLine storyLine.cpp)
target_link_libraries (storyLine glviewer)
