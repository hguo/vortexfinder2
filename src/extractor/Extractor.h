#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include "common/VortexObject.h"
#include "PuncturedElem.h"
#include <cmath>

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
  bool ExtractElem(ElemIdType id);

  void AddPuncturedFace(ElemIdType id, int f, int chirality, double pos[]);
  void AddPuncturedVirtualElemFace(FaceIdType id, int e, int chirality);
  
  virtual PuncturedElem* NewPuncturedElem(ElemIdType) const = 0;
  virtual PuncturedElem* NewPuncturedVirtualElem(FaceIdType) const = 0;

  virtual bool FindZero(const double X[][3], const double re[], const double im[], double pos[3]) const = 0;

protected:
  PuncturedElemMap _punctured_elems, 
                   _punctured_velems; // virtual elems
  std::vector<VortexObject> _vortex_objects;
  
  bool _gauge; 

protected:
  const GLDataset *_dataset;
}; 

#endif
