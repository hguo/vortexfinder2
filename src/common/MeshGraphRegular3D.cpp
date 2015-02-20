#include "MeshGraphRegular3D.h"
#include <cstring>
#include <climits>

MeshGraphRegular3D::MeshGraphRegular3D(int d_[3], bool pbc_[3])
{
  memcpy(d, d_, sizeof(int)*3);
  memcpy(pbc, pbc_, sizeof(bool)*3);
}

CCell MeshGraphRegular3D::Cell(CellIdType id) const
{
  CCell cell;
  int idx[3];
  
  cid2cidx(id, idx);
  if (!valid_cidx(idx)) return cell;
  const int i = idx[0], j = idx[1], k = idx[2];

  // nodes
  const int nodes_idx[8][3] = {
    {i, j, k}, {i+1, j, k}, {i+1, j+1, k}, {i, j+1, k},
    {i, j, k+1}, {i+1, j, k+1}, {i+1, j+1, k+1}, {i, j+1, k+1}};
  for (int i=0; i<8; i++) // don't worry about modIdx here. automatically done in idx2id()
    cell.nodes.push_back(nidx2nid(nodes_idx[i]));

  // faces
  const int faces_fidx[6][4] = {
    {i, j, k, 0}, // type0, yz
    {i, j, k, 1}, // type1, zx
    {i, j, k, 2}, // type2, xy
    {i+1, j, k, 0}, // type0, yz
    {i, j+1, k, 1}, // type1, zx
    {i, j, k+1, 2}};  // type2, xy
  const ChiralityType faces_chi[6] = {-1, -1, -1, 1, 1, 1};
  for (int i=0; i<6; i++) {
    cell.faces.push_back(fidx2fid(faces_fidx[i]));
    cell.faces_chirality.push_back(faces_chi[i]);
  }

  // neighbor cells
  const int neighbors_cidx[6][3] = { // need to be consistent with faces
    {i-1, j, k},
    {i, j-1, k},
    {i, j, k-1}, 
    {i+1, j, k},
    {i, j+1, k},
    {i, j, k+1}};
  for (int i=0; i<6; i++)
    if (valid_cidx(neighbors_cidx[i]))
      cell.neighbor_cells.push_back(cidx2cid(neighbors_cidx[i]));
    else
      cell.neighbor_cells.push_back(UINT_MAX);

  return cell;
}

CFace MeshGraphRegular3D::Face(FaceIdType id) const
{
  CFace face;
  int fidx[4];

  fid2fidx(id, fidx);
  if (!valid_fidx(fidx)) return face;
  const int i = fidx[0], j = fidx[1], k = fidx[2], t = fidx[3];

  // nodes
  const int nodes_idx[3][4][3] = {
    {{i, j, k}, {i, j+1, k}, {i, j+1, k+1}, {i, j, k+1}}, 
    {{i, j, k}, {i, j, k+1}, {i+1, j, k+1}, {i+1, j, k}},
    {{i, j, k}, {i+1, j, k}, {i+1, j+1, k}, {i, j+1, k}}};
  for (int i=0; i<4; i++)
    face.nodes.push_back(nidx2nid(nodes_idx[t][i]));

  // edges
  const int edges_idx[3][4][4] = {
    {{i, j, k, 1}, {i, j+1, k, 2}, {i, j, k+1, 1}, {i, j, k, 2}},
    {{i, j, k, 2}, {i, j, k+1, 0}, {i+1, j, k, 2}, {i, j, k, 0}},
    {{i, j, k, 0}, {i+1, j, k, 1}, {i, j+1, k, 0}, {i, j, k, 1}}};
  const ChiralityType edges_chi[4] = {1, 1, -1, -1};
  for (int i=0; i<4; i++) {
    face.edges.push_back(eidx2eid(edges_idx[t][i]));
    face.edges_chirality.push_back(edges_chi[i]);
  }

  // contained cells
  const int contained_cells_cidx[3][2][3] = {
    {{i, j, k}, {i-1, j, k}}, 
    {{i, j, k}, {i, j-1, k}},
    {{i, j, k}, {i, j, k-1}}};
  const ChiralityType contained_cells_chi[2] = {-1, 1};
  const int contained_cells_fid[3][2] = {
    {0, 3}, {1, 4}, {2, 5}};
  for (int i=0; i<2; i++) {
    face.contained_cells.push_back(cidx2cid(contained_cells_cidx[t][i]));
    face.contained_cells_chirality.push_back(contained_cells_chi[i]);
    face.contained_cells_fid.push_back(contained_cells_fid[t][i]);
  }

  return face;
}

CEdge MeshGraphRegular3D::Edge(EdgeIdType id) const
{
  CEdge edge;
  int eidx[4];

  eid2eidx(id, eidx);
  if (!valid_eidx(eidx)) return edge;
  const int i = eidx[0], j = eidx[1], k = eidx[2], t = eidx[3];

  // nodes
  const int nodes_idx[3][2][3] = {
    {{i, j, k}, {i+1, j, k}}, 
    {{i, j, k}, {i, j+1, k}}, 
    {{i, j, k}, {i, j, k+1}}};

  edge.node0 = id; // nidx2nid(node_idx[t][0]);
  edge.node1 = nidx2nid(nodes_idx[t][1]);

  // contained faces
  const int contained_faces_fidx[3][4][4] = {
    {{i, j, k, 2}, {i, j, k, 1}, {i, j-1, k, 2}, {i, j, k-1, 1}}, 
    {{i, j, k, 2}, {i, j, k, 0}, {i-1, j, k, 2}, {i, j, k-1, 0}},
    {{i, j, k, 1}, {i, j, k, 0}, {i-1, j, k, 1}, {i, j-1, k, 0}}};
  const ChiralityType contained_faces_chi[3][4] = {
    {1, -1, -1, 1}, {-1, 1, 1, -1}, {1, -1, -1, 1}};
  const int contained_faces_eid[3][4] = {
    {0, 3, 2, 1}, {3, 0, 1, 2}, {0, 3, 2, 1}};

  for (int i=0; i<4; i++) {
    edge.contained_faces.push_back(fidx2fid(contained_faces_fidx[t][i]));
    edge.contained_faces_chirality.push_back(contained_faces_chi[t][i]);
    edge.contained_faces_eid.push_back(contained_faces_eid[t][i]);
  }

  return edge;
}

EdgeIdType MeshGraphRegular3D::NEdges() const
{
  return NCells()*3;
}

EdgeIdType MeshGraphRegular3D::NFaces() const
{
  return NCells()*3;
}

EdgeIdType MeshGraphRegular3D::NCells() const
{
  return d[0]*d[1]*d[2];
}

void MeshGraphRegular3D::nid2nidx(unsigned int id, int idx[3]) const
{
  int s = d[0] * d[1]; 
  int k = id / s; 
  int j = (id - k*s) / d[0]; 
  int i = id - k*s - j*d[0]; 

  idx[0] = i; idx[1] = j; idx[2] = k;
}

void MeshGraphRegular3D::eid2eidx(unsigned int id, int idx[4]) const
{
  unsigned int nid = id / 3;
  nid2nidx(id, idx);
  idx[3] = id % 3;
}

void MeshGraphRegular3D::fid2fidx(unsigned int id, int idx[3]) const
{
  unsigned int nid = id / 3;
  nid2nidx(id, idx);
  idx[3] = id % 3;
}

void MeshGraphRegular3D::cid2cidx(unsigned int id, int idx[3]) const
{
  nid2nidx(id, idx);
}

unsigned int MeshGraphRegular3D::nidx2nid(const int idx_[3]) const
{
  int idx[3] = {idx_[0], idx_[1], idx_[2]};
  for (int i=0; i<3; i++) {
    idx[i] = idx[i] % d[i];
    if (idx[i] < 0)
      idx[i] += d[i];
  }
  return idx[0] + d[0] * (idx[1] + d[1] * idx[2]); 
}

unsigned int MeshGraphRegular3D::eidx2eid(const int idx[4]) const
{
  return nidx2nid(idx)*3 + idx[3];
}

unsigned int MeshGraphRegular3D::fidx2fid(const int idx[4]) const
{
  return nidx2nid(idx)*3 + idx[3];
}

unsigned int MeshGraphRegular3D::cidx2cid(const int idx[3]) const
{
  return nidx2nid(idx);
}

bool MeshGraphRegular3D::valid_nidx(const int idx[3]) const
{
  for (int i=0; i<3; i++) 
    if (idx[i]<0 || idx[i]>=d[i]) 
      return false;
  return true;
}

bool MeshGraphRegular3D::valid_eidx(const int eidx[4]) const
{
  if (eidx[3]<0 || eidx[3]>=3) return false;
  else return valid_cidx(eidx);
}

bool MeshGraphRegular3D::valid_fidx(const int fidx[4]) const
{
  if (fidx[3]<0 || fidx[3]>=3) return false;
  else return valid_cidx(fidx);
}

bool MeshGraphRegular3D::valid_cidx(const int idx[3]) const
{
  for (int i=0; i<3; i++)
    if (pbc[i]) {
      if (idx[i] >= d[i]) return false;
    } else {
      if (idx[i] >= d[i]-1) return false;
    }
  return true;
}

