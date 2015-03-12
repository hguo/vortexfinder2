if (WITH_VTK)
  include (${VTK_USE_FILE})
endif ()

if (WITH_QT)
  include (${QT_USE_FILE})
  include_directories (
    # ${QT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
  )
endif ()

set (viewer_sources
  trackball.cpp
  widget.cpp
  storyLineWidget.cpp
  mainWindow.cpp)

qt4_wrap_ui (viewer_headers 
  mainWindow.ui)

if (APPLE) # remove anoying OpenGL deprecated warnings
  add_definitions ("-Wno-deprecated")
endif ()

add_library (glviewer ${viewer_sources} ${viewer_headers})
set_target_properties (glviewer PROPERTIES AUTOMOC ON)
target_link_libraries (glviewer 
  glcommon
  glio
  ${QT_LIBRARIES}
  ${GLUT_LIBRARY}
  ${GLEW_LIBRARY}
  ${OPENGL_LIBRARIES}
)

if (WITH_VTK)
  target_link_libraries (glviewer ${VTK_LIBRARIES})
endif ()

add_executable (viewer main.cpp)
target_link_libraries (viewer glviewer)

add_executable (viewer1 main1.cpp)
target_link_libraries (viewer1 glviewer)

add_executable (storyLine storyLine.cpp)
target_link_libraries (storyLine glviewer)
