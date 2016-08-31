if (WITH_LIBMESH)
  add_executable (extractor_condor2 ex_condor2.cpp)
  target_link_libraries (extractor_condor2 PUBLIC glextractor)
endif ()

add_executable (extractor_glgpu3D ex_glgpu3D.cpp)
target_link_libraries (extractor_glgpu3D PUBLIC glextractor)

add_executable (extractor_glgpu3D_stream ex_glgpu3D_stream.cpp)
target_link_libraries (extractor_glgpu3D_stream PUBLIC glextractor)

if (WITH_TBB)
  add_executable (extractor_glgpu3D_stream_tbb ex_glgpu3D_stream_tbb.cpp)
  target_link_libraries (extractor_glgpu3D_stream_tbb PUBLIC glextractor ${TBB_LIBRARY_RELEASE})
  if (WITH_LEVELDB)
    target_link_libraries (extractor_glgpu3D_stream_tbb PUBLIC glextractor ${LevelDB_LIBRARY})
  endif ()
endif ()

# add_executable (extractor_glgpu3D_sto ex_glgpu3D_sto.cpp)
# target_link_libraries (extractor_glgpu3D_sto PUBLIC glextractor)

# add_executable (extractor_glgpu3D_box ex_glgpu3D_box.cpp)
# target_link_libraries (extractor_glgpu3D_box PUBLIC glextractor)

add_executable (extractor_glgpu2D ex_glgpu2D.cpp)
target_link_libraries (extractor_glgpu2D PUBLIC glextractor)

# add_executable (extractor_presave ex_presave.cpp)
# target_link_libraries (extractor_presave PUBLIC glextractor)

add_executable (events ex_events.cpp)
target_link_libraries (events glcommon)

add_executable (events_dist ex_events_dist.cpp)
target_link_libraries (events_dist glcommon)

add_executable (dist ex_dist.cpp)
target_link_libraries (dist glio)

# add_executable (count ex_count.cpp)
# target_link_libraries (count glio)

add_executable (tracer_glgpu3D ex_tracer_glgpu.cpp)
target_link_libraries (tracer_glgpu3D PUBLIC gltracer glextractor)
