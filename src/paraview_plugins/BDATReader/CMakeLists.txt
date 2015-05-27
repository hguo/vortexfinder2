add_paraview_plugin (
  BDATReader "0.1"
  SERVER_MANAGER_XML bdat.xml
  SERVER_MANAGER_SOURCES vtkBDATReader.cxx
)

set (CMAKE_INSTALL_NAME_DIR "@executable_path/../lib")

target_link_libraries (BDATReader glio)
install (TARGETS BDATReader DESTINATION lib)
