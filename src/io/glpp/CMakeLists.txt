set (glpp_source
  GL_post_process.cpp
  datafile.cpp
  fileutils.cpp
  paramfile.cpp
  stringutils.cpp
)

add_library (glpp STATIC ${glpp_source})
