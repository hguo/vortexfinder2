add_executable (test_bdat test_bdat.cpp)
target_link_libraries (test_bdat glio)

add_executable (test_glgpudata test_glgpudata.cpp)
target_link_libraries (test_glgpudata glio)

add_executable (test_condor2data test_condor2data.cpp)
target_link_libraries (test_condor2data glio)
