set (glpp_source
  GL_post_process.cpp
  datafile.cpp
  fileutils.cpp
  paramfile.cpp
  stringutils.cpp
)

add_library (glpp STATIC ${glpp_source})

add_paraview_plugin (
  GLGPUSupercurrentFilter "0.1"
  SERVER_MANAGER_XML glgpu_supercurrent.xml
  SERVER_MANAGER_SOURCES vtkGLGPUSupercurrentFilter.cxx
)

target_link_libraries (GLGPUSupercurrentFilter PRIVATE glpp)
