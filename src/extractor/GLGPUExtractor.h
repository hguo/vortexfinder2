#ifndef _GLGPUEXTRACTOR_H
#define _GLGPUEXTRACTOR_H

#include <map>
#include <list>
#include "Extractor.h"

class GLGPUVortexExtractor : public VortexExtractor {
public:
  GLGPUVortexExtractor(); 
  ~GLGPUVortexExtractor();

  void SetDataset(const GLDataset *ds); 

  void Extract();

private:
  void ExtractElem(int *idx);

  double gauge(int *x0, int *x1) const;

private: 
  const GLGPUDataset *_ds; 
}; 

#endif
