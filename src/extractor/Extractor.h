#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include "io/GLDataset.h"

class VortexExtractor {
public: 
  VortexExtractor(); 
  ~VortexExtractor(); 

  virtual void SetDataset(const GLDataset* ds) = 0;
}; 

#endif
