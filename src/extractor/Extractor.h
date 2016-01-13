#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H

#include "common/VortexLine.h"
#include "common/VortexObject.h"
#include "common/VortexTransition.h"
#include "common/Puncture.h"
#include "InverseInterpolation.h"
#include <map>

class GLDataset;
class GLDatasetBase;

enum {
  INTERPOLATION_TRI_CENTER = 0x1,
  INTERPOLATION_TRI_BARYCENTRIC = 0x10, 
  INTERPOLATION_QUAD_CENTER = 0x100,
  INTERPOLATION_QUAD_BARYCENTRIC = 0x1000,
  INTERPOLATION_QUAD_BILINEAR = 0x1000, 
  INTERPOLATION_QUAD_LINECROSS = 0x10000
}; 

class VortexExtractor {
public: 
  VortexExtractor(); 
  ~VortexExtractor(); 

  void SetNumberOfThreads(int);
  void SetInterpolationMode(unsigned int);

  void SetGaugeTransformation(bool);
  void SetArchive(bool); // archive intermediate results for data reuse
  void SetGPU(bool);
  
  virtual void SetDataset(const GLDatasetBase* ds);
  const GLDataset* Dataset() const {return (GLDataset*)_dataset;}

  void ExtractFaces(int slot=0);
  void ExtractFaces(std::vector<FaceIdType> faces, int slot, int &positive, int &negative);
  void ExtractEdges();
  
  void ExtractFaces_GPU(int slot=0);
  void ExtractEdges_GPU();

  bool SavePuncturedEdges() const;
  bool LoadPuncturedEdges();
  bool SavePuncturedFaces(int slot=0) const; 
  bool LoadPuncturedFaces(int slot=0);
  void ClearPuncturedObjects();
  
  void SaveVortexLines(int slot=0);
  std::vector<VortexLine> GetVortexLines(int slot=0);

  void TraceVirtualCells();
  void TraceOverSpace(int slot=0);
  void TraceOverTime();
  void AnalyzeTransition();

  void RelateOverTime();

  void RotateTimeSteps();

public:
  int ExtractFace(FaceIdType, int slot=0); // returns chirality
  void ExtractSpaceTimeEdge(EdgeIdType);

protected:
  void VortexObjectsToVortexLines(const std::map<FaceIdType, PuncturedFace>& pfs, const std::vector<VortexObject>& vobjs, std::vector<VortexLine>& vlines, bool bezier=false);
  int NewGlobalVortexId();
  void ResetGlobalVortexId();

protected:
  void AddPuncturedFace(FaceIdType, int slot, ChiralityType chirality, const double pos[3]);
  void AddPuncturedEdge(EdgeIdType, ChiralityType chirality, double t);

  bool FindFaceZero(int n, const double X[][3], const double re[], const double im[], double pos[3]) const;
  bool FindSpaceTimeEdgeZero(const double re[], const double im[], double &t) const;

protected:
  std::map<FaceIdType, PuncturedFace> _punctured_faces, _punctured_faces1; 
  std::map<CellIdType, PuncturedCell> _punctured_cells, _punctured_cells1;
  std::map<EdgeIdType, PuncturedEdge> _punctured_edges;
  // std::map<FaceIdType, PuncturedCell> _punctured_vcells;
  std::map<FaceIdType, std::vector<FaceIdType> > _related_faces;

  std::vector<VortexObject> _vortex_objects, _vortex_objects1;
  std::vector<VortexLine> _vortex_lines, _vortex_lines1;

  VortexTransition _vortex_transition;

protected:
  const GLDatasetBase *_dataset;
  bool _gauge; 
  bool _archive;
  bool _gpu;
  unsigned int _interpolation_mode;

private:
  static void *execute_thread_helper(void *ctx);
  void execute_thread(int nthreads, int tid, int type, int slot);

  int _nthreads;
}; 

#endif
