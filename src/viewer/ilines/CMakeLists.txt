set (ilines_sources
  ILRender.cpp  
  ILTexture.cpp  
  ILUtilities.cpp  
  ShaderProgram.cpp  
  Vector.cpp
  shaders/IL_cylinder_blinn_fp.cpp          
  shaders/IL_cylinder_blinn_vp.cpp
  shaders/IL_cylinder_blinn_tangent_vp.cpp
)

add_library (ilines ${ilines_sources})
target_link_libraries (ilines ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
