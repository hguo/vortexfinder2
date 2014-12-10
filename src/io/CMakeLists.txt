set (io_sources
  GLGPUDataset.cpp
  Condor2Dataset.cpp
  GLDataset.cpp
)

add_library (glio ${io_sources})
target_link_libraries (glio glcommon ${LIBMESH_LIBRARY})
