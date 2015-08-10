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
  _mg = new MeshGraphRegular2D(_h[0].dims, _h[0].pbc);
}

CellIdType GLGPU2DDataset::Pos2CellId(const double X[]) const
{
  // TODO
  return false;
}
