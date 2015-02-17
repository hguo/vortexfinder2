#ifndef _GLGPU3DEXTRACTOR_H
#define _GLGPU3DEXTRACTOR_H

#include <map>
#include <list>
#include "Extractor.h"

enum {
  INTERPOLATION_CENTER = 0, 
  INTERPOLATION_BARYCENTRIC, 
  INTERPOLATION_BILINEAR, 
  INTERPOLATION_LINECROSS
}; 

class GLGPU3DVortexExtractor : public VortexExtractor {
public:
  GLGPU3DVortexExtractor(); 
  ~GLGPU3DVortexExtractor();

  void SetInterpolationMode(int);

  void Extract();

protected:
  bool FindFaceZero(const double X[][3], const double re[], const double im[], double pos[3]) const;

private: 
  int _interpolation_mode;
}; 

#endif
