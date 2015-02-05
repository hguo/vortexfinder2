#ifndef _GLGPUEXTRACTOR_H
#define _GLGPUEXTRACTOR_H

#include <map>
#include <list>
#include "Extractor.h"

enum {
  INTERPOLATION_CENTER = 0, 
  INTERPOLATION_BARYCENTRIC, 
  INTERPOLATION_BILINEAR, 
  INTERPOLATION_LINECROSS
}; 

class GLGPUVortexExtractor : public VortexExtractor {
public:
  GLGPUVortexExtractor(); 
  ~GLGPUVortexExtractor();

  void SetInterpolationMode(int);

  void Extract();

protected:
  PuncturedElem* NewPuncturedElem(ElemIdType) const;
  PuncturedElem* NewPuncturedPrism(FaceIdType) const;
  
  bool FindZero(const double X[][3], const double re[], const double im[], double pos[3]) const;

private: 
  int _interpolation_mode;
}; 

#endif
