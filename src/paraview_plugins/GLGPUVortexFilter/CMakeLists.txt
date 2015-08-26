add_paraview_plugin (
  GLGPUVortexFilter "0.1"
  SERVER_MANAGER_XML glgpu_vortex.xml
  SERVER_MANAGER_SOURCES vtkGLGPUVortexFilter.cxx
)

target_link_libraries (GLGPUVortexFilter PRIVATE glextractor glcommon glio)
