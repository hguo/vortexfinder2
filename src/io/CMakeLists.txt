set (io_sources
  GLGPUDataset.cpp
  GLDataset.cpp
)
  
if (WITH_LIBMESH)
  set (io_sources ${io_sources}
    Condor2Dataset.cpp)
endif ()

add_library (glio ${io_sources})

if (WITH_LIBMESH)
  target_link_libraries (glio glcommon ${LIBMESH_LIBRARY} ${NETCDF_LIBRARY})
else ()
  target_link_libraries (glio glcommon)
endif ()
