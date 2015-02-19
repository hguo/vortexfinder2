#include "GLGPU2DDataset.h"
#include "common/MeshGraphRegular2D.h"

GLGPU2DDataset::GLGPU2DDataset()
{
}

GLGPU2DDataset::~GLGPU2DDataset()
{
}
  
void GLGPU2DDataset::SerializeDataInfoToString(std::string& buf) const
{
}

void GLGPU2DDataset::BuildMeshGraph()
{
  if (_mg != NULL) delete _mg;
  _mg = new MeshGraphRegular2D(_dims, _pbc);
}
  
void GLGPU2DDataset::GetFaceValues(const CFace&, int time, double X[][3], double A[][3], double re[], double im[]) const
{
  // TODO
}

void GLGPU2DDataset::GetSpaceTimeEdgeValues(const CEdge&, double X[][3], double A[][3], double re[], double im[]) const
{
  // TODO
}

bool GLGPU2DDataset::Psi(const double X[2], double &re, double &im) const
{
  // TODO
  return false;
}

bool GLGPU2DDataset::Supercurrent(const double X[2], double J[3]) const
{
  // TODO
  return false;
}

CellIdType GLGPU2DDataset::Pos2CellId(const double X[]) const
{
  // TODO
  return false;
}
