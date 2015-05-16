add_paraview_plugin (
  BDATReader "0.1"
  SERVER_MANAGER_XML bdat.xml
  SERVER_MANAGER_SOURCES vtkBDATReader.cxx
)

target_link_libraries (BDATReader glio)
