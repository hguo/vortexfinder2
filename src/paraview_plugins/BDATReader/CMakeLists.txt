add_paraview_plugin (
  BDATReader "0.1"
  SERVER_MANAGER_XML BDATReader.xml
  SERVER_MANAGER_SOURCES vtkBDATReader.cxx
)

target_link_libraries (BDATReader PRIVATE glio)
