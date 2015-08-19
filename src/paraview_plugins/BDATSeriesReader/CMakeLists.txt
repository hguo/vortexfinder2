add_paraview_plugin (
  BDATSeriesReader "0.1"
  SERVER_MANAGER_XML BDATSeriesReader.xml
  SERVER_MANAGER_SOURCES vtkBDATSeriesReader.cxx
)

target_link_libraries (BDATSeriesReader PRIVATE glio)
