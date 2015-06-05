add_paraview_plugin (
  BDATSeriesReader "0.1"
  SERVER_MANAGER_XML BDATSeriesReader.xml
  SERVER_MANAGER_SOURCES vtkBDATSeriesReader.cxx
)

set (CMAKE_INSTALL_NAME_DIR "@executable_path/../lib")

target_link_libraries (BDATSeriesReader glio)
install (TARGETS BDATSeriesReader DESTINATION lib)
