#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include "common/VortexObject.h"
#include "PuncturedElem.h"

class GLDataset;

class VortexExtractor {
public: 
  VortexExtractor(); 
  ~VortexExtractor(); 

  virtual void SetDataset(const GLDataset* ds);
  void SetGaugeTransformation(bool); 

  virtual void Extract() = 0; 
  void Trace(); 

  void WriteVortexObjects(const std::string& filename); 

protected:
  PuncturedElemMap _punctured_elems; 
  std::vector<VortexObject> _vortex_objects;
  
  bool _gauge; 

protected:
  const GLDataset *_dataset;
}; 

#endif
