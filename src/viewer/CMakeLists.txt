if (WITH_VTK)
  find_package(VTK REQUIRED
    COMPONENTS
      CommonCore
      CommonDataModel
      FiltersCore
      IOImage
      RenderingCore
      RenderingOpenGL2)
endif ()

if (WITH_QT)
  include (${QT_USE_FILE})
  include_directories (
    # ${QT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
  )
endif ()

set (viewer_sources
  trackball.cpp
  widget.cpp
  # storyLineWidget.cpp
  mainWindow.cpp)

if (WITH_FORTRAN)
  list (APPEND viewer_sources cmds2.f90)
endif ()

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
  gl2ps
  ilines
  ${QT_LIBRARIES}
  ${GLUT_LIBRARY}
  ${GLEW_LIBRARY}
  ${OPENGL_LIBRARIES}
)

if (WITH_VTK)
  target_link_libraries (glviewer ${VTK_LIBRARIES})
endif ()

if (WITH_CUDA)
  # target_link_libraries (glviewer glvolren)
endif ()

if (WITH_FORTRAN)
  target_link_libraries (glviewer ${LAPACK_LIBRARIES})
endif ()

# add_executable (viewer main.cpp)
# target_link_libraries (viewer glviewer)

add_executable (viewer1 main1.cpp)
target_link_libraries (viewer1 glviewer)

if (WITH_ROCKSDB)
  add_executable (viewer2 main2.cpp)
  target_link_libraries (viewer2 glviewer)
endif ()

# add_executable (storyLine storyLine.cpp)
# target_link_libraries (storyLine glviewer)

# subdirs (volren gl2ps ilines)
subdirs (gl2ps ilines)
