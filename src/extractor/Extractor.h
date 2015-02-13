#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include "common/VortexObject.h"
#include "InverseInterpolation.h"
#include "Puncture.h"
#include <map>

class GLDataset;
class GLDatasetBase;

class VortexExtractor {
public: 
  VortexExtractor(); 
  ~VortexExtractor(); 

  void SetGaugeTransformation(bool); 
  
  virtual void SetDataset(const GLDatasetBase* ds);
  const GLDataset* Dataset() const {return (GLDataset*)_dataset;}

  virtual void Extract() {}; 
  void Trace();

  void WriteVortexObjects(const std::string& filename); 

  bool SavePuncturedEdges() const;
  bool LoadPuncturedEdges();
  bool SavePuncturedFaces(int time) const; 
  bool LoadPuncturedFaces(int time);
  void ClearPuncturedObjects();

  void TraceVirtualCells();

protected:
  void AddPuncturedFace(FaceIdType, int time, int chirality, const double pos[3]);
  void AddPuncturedEdge(EdgeIdType, int chirality, double t);

  virtual bool FindFaceZero(const double X[][3], const double re[], const double im[], double pos[3]) const {return false;}
  bool FindSpaceTimeEdgeZero(const double re[], const double im[], double &t) const;

protected:
  std::map<FaceIdType, PuncturedFace> _punctured_faces, _punctured_faces1; 
  std::map<EdgeIdType, PuncturedEdge> _punctured_edges;
  std::map<FaceIdType, PuncturedCell> _punctured_cells, _punctured_vcells;

  std::vector<VortexObject> _vortex_objects;
  
  bool _gauge; 

protected:
  const GLDatasetBase *_dataset;
}; 

#endif
