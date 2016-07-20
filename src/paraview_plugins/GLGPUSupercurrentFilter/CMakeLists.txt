add_paraview_plugin (
  GLGPUSupercurrentFilter "0.1"
  SERVER_MANAGER_XML glgpu_supercurrent.xml
  SERVER_MANAGER_SOURCES vtkGLGPUSupercurrentFilter.cxx
)

target_link_libraries (GLGPUSupercurrentFilter PRIVATE glpp)
