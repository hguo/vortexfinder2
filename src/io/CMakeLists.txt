set (io_sources
  BDATReader.cpp
  GLGPUDataset.cpp
  GLDataset.cpp
  MeshGraph.cpp
)
  
if (WITH_LIBMESH)
  set (io_sources ${io_sources}
    # MeshWrapper.cpp
    Condor2Dataset.cpp)
endif ()

add_library (glio ${io_sources})

if (WITH_LIBMESH)
  target_link_libraries (glio glcommon ${LIBMESH_LIBRARY} ${NETCDF_LIBRARY})
else ()
  target_link_libraries (glio glcommon)
endif ()

add_executable (test_meshgraph test_meshgraph.cpp)
target_link_libraries (test_meshgraph glio)
