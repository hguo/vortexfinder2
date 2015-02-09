#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include "common/VortexObject.h"
#include "InverseInterpolation.h"
#include "Puncture.h"
#include <map>

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
  void AddPuncturedFace(FaceIdType, int time, int chirality, const double pos[3]);
  void AddPuncturedEdge(EdgeIdType, int chirality, double t);

  virtual bool FindFaceZero(const double X[][3], const double re[], const double im[], double pos[3]) const = 0;
  bool FindSpaceTimeEdgeZero(const double re[], const double im[], double &t) const;

protected:
  std::map<FaceIdType, PuncturedFace> _punctured_faces, _punctured_faces1; 
  std::map<EdgeIdType, PuncturedEdge> _punctured_edges;

  std::vector<VortexObject> _vortex_objects;
  
  bool _gauge; 

protected:
  const GLDataset *_dataset;
}; 

#endif
