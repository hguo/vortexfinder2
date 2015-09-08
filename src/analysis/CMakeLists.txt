add_executable (events ex_events.cpp)
target_link_libraries (events glcommon)

add_executable (dist ex_dist.cpp)
target_link_libraries (dist glio)
