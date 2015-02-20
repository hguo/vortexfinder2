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
  void GetFaceValues(const CFace&, int time, double X[][3], double A[][3], double re[], double im[]) const;
  void GetSpaceTimeEdgeValues(const CEdge&, double X[][3], double A[][3], double re[], double im[]) const;

public:
  bool Pos(NodeIdType, double X[2]) const;
  bool Psi(const double X[2], double &re, double &im, int slot=0) const;
  bool Psi(NodeIdType, double &re, double &im, int slot=0) const;
  bool Supercurrent(const double X[2], double J[3], int slot=0) const;
  bool Supercurrent(NodeIdType, double J[3], int slot=0) const;
  
  CellIdType Pos2CellId(const double X[]) const; 
};

#endif
