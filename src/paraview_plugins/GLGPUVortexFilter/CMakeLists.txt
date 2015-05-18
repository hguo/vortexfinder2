add_paraview_plugin (
  GLGPUVortex "0.1"
  SERVER_MANAGER_XML glgpu_vortex.xml
  SERVER_MANAGER_SOURCES vtkGLGPUVortexFilter.cxx
)

target_link_libraries (GLGPUVortex glextractor)
