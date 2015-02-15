#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include "common/VortexLine.h"
#include "common/VortexObject.h"
#include "common/Puncture.h"
#include "InverseInterpolation.h"
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

  void SaveVortexLines(const std::string& filename); 

  bool SavePuncturedEdges() const;
  bool LoadPuncturedEdges();
  bool SavePuncturedFaces(int time) const; 
  bool LoadPuncturedFaces(int time);
  void ClearPuncturedObjects();

  void TraceVirtualCells();
  void TraceOverSpace(int time);
  void TraceOverTime();

  void RelateOverTime();

  void PrepareForNextStep();

protected:
  void VortexObjectsToVortexLines(const std::map<FaceIdType, PuncturedFace>& pfs, const std::vector<VortexObject>& vobjs, std::vector<VortexLine>& vlines);
  int NewVortexId();

protected:
  void AddPuncturedFace(FaceIdType, int time, ChiralityType chirality, const double pos[3]);
  void AddPuncturedEdge(EdgeIdType, ChiralityType chirality, double t);

  virtual bool FindFaceZero(const double X[][3], const double re[], const double im[], double pos[3]) const {return false;}
  bool FindSpaceTimeEdgeZero(const double re[], const double im[], double &t) const;

protected:
  std::map<FaceIdType, PuncturedFace> _punctured_faces, _punctured_faces1; 
  std::map<CellIdType, PuncturedCell> _punctured_cells, _punctured_cells1;
  std::map<EdgeIdType, PuncturedEdge> _punctured_edges;
  std::map<FaceIdType, PuncturedCell> _punctured_vcells;
  std::map<FaceIdType, std::vector<FaceIdType> > _related_faces;

  std::vector<VortexObject> _vortex_objects, _vortex_objects1;
  std::vector<VortexLine> _vortex_lines;
 
  int _num_vortices;

protected:
  const GLDatasetBase *_dataset;
  bool _gauge; 
}; 

#endif
