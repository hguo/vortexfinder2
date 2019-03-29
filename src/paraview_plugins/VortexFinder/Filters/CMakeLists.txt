set(classes
  vtkBDATReader
  vtkBDATSeriesReader
  vtkGLGPUSupercurrentFilter
  vtkGLGPUVortexFilter)

vtk_module_add_module(VortexFilters
  CLASSES ${classes})
vtk_module_link(VortexFilters
  PRIVATE
    glcommon
    glextractor
    glio
    glpp)

paraview_add_server_manager_xmls(
  XMLS
    BDATReader.xml
    BDATSeriesReader.xml
    glgpu_supercurrent.xml
    glgpu_vortex.xml)
