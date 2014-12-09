#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include "io/GLDataset.h"
#include "vortex/VortexObject.h"

class VortexExtractor {
public: 
  VortexExtractor(); 
  ~VortexExtractor(); 

  // virtual void Clear() = 0; 
  virtual void SetDataset(const GLDataset* ds) = 0;

  void WriteVortexObjects(const std::string& filename); 
  
protected:
  std::vector<VortexObject> _vortex_objects; 
}; 

#endif
