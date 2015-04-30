#ifndef _GLGPU2DDATASET_H
#define _GLGPU2DDATASET_H

#include "io/GLGPUDataset.h"

class GLGPU2DDataset : public GLGPUDataset
{
public:
  GLGPU2DDataset();
  ~GLGPU2DDataset();

  void SerializeDataInfoToString(std::string& buf) const;

  void BuildMeshGraph();

public:
  int Dimensions() const {return 2;}
  int NrFacesPerCell() const {return 1;}
  int NrNodesPerFace() const {return 4;}

public:
  CellIdType Pos2CellId(const double X[]) const; 
};

#endif
