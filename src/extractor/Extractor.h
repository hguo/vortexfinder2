#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include "io/GLDataset.h"
#include "common/VortexObject.h"

class VortexExtractor {
public: 
  VortexExtractor(); 
  ~VortexExtractor(); 

  virtual void SetDataset(const GLDataset* ds) = 0;
  void SetGaugeTransformation(bool); 

  virtual void Extract() = 0; 
  virtual void Trace() = 0; 

  void WriteVortexObjects(const std::string& filename); 
  
protected:
  std::vector<VortexObject> _vortex_objects;
  bool _gauge; 
}; 

#endif
