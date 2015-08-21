#ifndef _GLGPUEXTRACTOR_H
#define _GLGPUEXTRACTOR_H

#include <map>
#include <list>
#include "Extractor.h"

enum {
  INTERPOLATION_QUAD_CENTER = 0, 
  INTERPOLATION_QUAD_BARYCENTRIC,
  INTERPOLATION_QUAD_BILINEAR, 
  INTERPOLATION_QUAD_LINECROSS,
  INTERPOLATION_TRI_BARYCENTRIC, 
  INTERPOLATION_TRI_CENTER 
}; 

class GLGPUVortexExtractor : public VortexExtractor {
public:
  GLGPUVortexExtractor(); 
  ~GLGPUVortexExtractor();

  void SetInterpolationMode(int);

protected:
  bool FindFaceZero(const double X[][3], const double re[], const double im[], double pos[3]) const;

private: 
  int _interpolation_mode;
}; 

#endif
