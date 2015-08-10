add_executable (test_bdat test_bdat.cpp)
target_link_libraries (test_bdat glio)

add_executable (test_glgpudata test_glgpudata.cpp)
target_link_libraries (test_glgpudata glio)

add_executable (test_condor2data test_condor2data.cpp)
target_link_libraries (test_condor2data glio)

add_executable (test_meshgraph test_meshgraph.cpp)
target_link_libraries (test_meshgraph glio)

add_executable (test_curves test_curves.cpp)

add_executable (test_inclusions test_inclusions.cpp)
target_link_libraries (test_inclusions glcommon)

#add_executable (test_nc test_nc.cpp)
#target_link_libraries (test_nc glio)
