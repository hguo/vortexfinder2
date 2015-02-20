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
  
void GLGPU2DDataset::GetFaceValues(const CFace& f, int slot, double X[][3], double A_[][3], double re[], double im[]) const
{
  for (int i=0; i<f.nodes.size(); i++) {
    Pos(f.nodes[i], X[i]);
    // A(X[i], A_[i], slot);
    A(f.nodes[i], A_[i], slot);
    Psi(f.nodes[i], re[i], im[i], slot);
  }
}

void GLGPU2DDataset::GetSpaceTimeEdgeValues(const CEdge& e, double X[][3], double A_[][3], double re[], double im[]) const
{
  Pos(e.node0, X[0]);
  Pos(e.node1, X[1]);

  A(e.node0, A_[0], 0);
  A(e.node1, A_[1], 0);
  A(e.node1, A_[2], 1);
  A(e.node0, A_[3], 1);

  Psi(e.node0, re[0], im[0], 0);
  Psi(e.node1, re[1], im[1], 0);
  Psi(e.node1, re[2], im[2], 1);
  Psi(e.node0, re[3], im[3], 1);

#if 0
  fprintf(stderr, "re={%f, %f, %f, %f}, im={%f, %f, %f, %f}\n",
      re[0], re[1], re[2], re[3],
      im[0], im[1], im[2], im[3]);
#endif
}

bool GLGPU2DDataset::Pos(NodeIdType id, double X[3]) const
{
  int idx[3];

  Nid2Idx(id, idx);
  Idx2Pos(idx, X);

  return true;
}

bool GLGPU2DDataset::Psi(const double X[3], double &re, double &im, int slot) const
{
  // TODO
  return false;
}

bool GLGPU2DDataset::Psi(NodeIdType id, double &re, double &im, int slot) const
{
  double *r = slot == 0 ? _re : _re1;
  double *i = slot == 0 ? _im : _im1;
  // fprintf(stderr, "%p, %p\n", r, i);

  re = r[id]; 
  im = i[id];

  return true;
}

bool GLGPU2DDataset::Supercurrent(const double X[2], double J[3], int slot) const
{
  // TODO
  return false;
}

bool GLGPU2DDataset::Supercurrent(NodeIdType, double J[3], int slot) const
{
  // TODO
  return false;
}

CellIdType GLGPU2DDataset::Pos2CellId(const double X[]) const
{
  // TODO
  return false;
}
