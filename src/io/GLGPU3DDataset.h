#ifndef _GLGPU3D_DATASET_H
#define _GLGPU3D_DATASET_H

#include <cmath>
#include "GLGPUDataset.h"
#include "common/Texel.hpp"

class GLGPU3DDataset : public GLGPUDataset
{
public: 
  GLGPU3DDataset(); 
  ~GLGPU3DDataset();
  
  // void ComputeSupercurrentField(int slot=0);

public: // mesh graph
  void SetMeshType(int); // hex or tet
  int MeshType() const {return _mesh_type;}

  void BuildMeshGraph();
  
  std::vector<FaceIdType> GetBoundaryFaceIds(int type) const; // 0: YZ, 1: ZX, 2: XY

public: // mesh info
  int Dimensions() const {return 3;}
  
public: // mesh utils
  CellIdType Pos2CellId(const float X[]) const; 

public: // data access
#if 0
  const float& Re(int x, int y, int z) const {return texel3D(_re, _dims, x, y, z);}
  const float& Im(int x, int y, int z) const {return texel3D(_im, _dims, x, y, z);}
  float Rho(int x, int y, int z) const {float r=Re(x, y, z), i=Im(x, y, z); return sqrt(r*r+i*i);}
  float Phi(int x, int y, int z) const {float r=Re(x, y, z), i=Im(x, y, z); return atan2(i, r);}
#endif
  
  // Order parameters (direct access/linear interpolation)
  bool Psi(const float X[3], float &re, float &im) const;

  // Supercurrent field
  // bool Supercurrent(const float X[3], float J[3]) const;

private:
  bool _mesh_type;
}; 

#endif
