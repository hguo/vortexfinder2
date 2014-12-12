#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include "io/GLDataset.h"
#include "common/VortexObject.h"
#include "PuncturedElem.h"

class VortexExtractor {
public: 
  VortexExtractor(); 
  ~VortexExtractor(); 

  virtual void SetDataset(const GLDataset* ds) = 0;
  void SetGaugeTransformation(bool); 

  virtual void Extract() = 0; 
  void Trace(); 

  void WriteVortexObjects(const std::string& filename); 

protected:
  virtual std::vector<unsigned int> Neighbors(unsigned int elem_id) const = 0;

protected:
  PuncturedElemMap _punctured_elems; 
  std::vector<VortexObject> _vortex_objects;
  
  bool _gauge; 
}; 

#endif
