#ifndef _MESHGRAPH_H
#define _MESHGRAPH_H

#include <vector>
#include <bitset>
#include <map>
#include "def.h"

struct CEdge;
struct CFace;
struct CCell;

typedef std::tuple<NodeIdType, NodeIdType> EdgeIdType2;
typedef std::tuple<NodeIdType, NodeIdType, NodeIdType> FaceIdType3;
typedef std::tuple<NodeIdType, NodeIdType, NodeIdType, NodeIdType> FaceIdType4;

EdgeIdType2 AlternateEdge(EdgeIdType2 e, int chirality);
FaceIdType3 AlternateFace(FaceIdType3 f, int rotation, int chirality);
FaceIdType4 AlternateFace(FaceIdType4 f, int rotation, int chirality);

struct CEdge {
  // nodes
  NodeIdType node0, node1;

  // neighbor faces (unordered)
  std::vector<const CFace*> neighbor_faces;
  std::vector<int> neighbor_face_chiralities;
  std::vector<int> neighbor_face_eid; // the edge id in the corresponding face
};

struct CFace {
  // nodes (ordered)
  std::vector<NodeIdType> nodes;

  // edges (ordered)
  std::vector<CEdge*> edges;
  std::vector<int> edge_chiralities;

  // neighbor cells, only two, chirality(cell0)=-1, chirality(cell1)=1
  const CCell *neighbor_cell0, *neighbor_cell1;
  int neighbor_cell0_fid, neighbor_cell1_fid;

  // utils
  bool on_boundary() const {return neighbor_cell0 == NULL || neighbor_cell1 == NULL;}
};

struct CCell {
  // nodes (ordered)
  std::vector<NodeIdType> nodes;

  // faces (ordered)
  std::vector<const CFace*> faces;
  std::vector<int> face_chiralities;

  // neighbor cells (ordered)
  std::vector<const CCell*> neighbor_cells;
};

struct MeshGraph {
  std::vector<CEdge*> edges;
  std::vector<CFace*> faces;
  std::vector<CCell*> cells;

  ~MeshGraph();
};


class MeshGraphBuilder {
public:
  explicit MeshGraphBuilder(CellIdType n_cells, MeshGraph& mg);
  virtual ~MeshGraphBuilder() {}
 
  virtual void Build() = 0;

protected:
  MeshGraph &_mg;
};

class MeshGraphBuilder_Tet : public MeshGraphBuilder {
public:
  explicit MeshGraphBuilder_Tet(CellIdType n_cells, MeshGraph& mg) : MeshGraphBuilder(n_cells, mg) {}
  ~MeshGraphBuilder_Tet() {}

  CCell* AddCell(CellIdType i, 
      const std::vector<NodeIdType> &nodes, 
      const std::vector<CellIdType> &neighbors, 
      const std::vector<FaceIdType3> &faces);

  void Build();

private:
  CEdge* AddEdge(EdgeIdType2 e, int &chirality, const CFace* f, int eid);
  CFace* AddFace(FaceIdType3 f, int &chirality, const CCell* el, int fid);
  CEdge* GetEdge(EdgeIdType2 e, int &chirality);
  CFace* GetFace(FaceIdType3 f, int &chirality);

private:
  std::map<EdgeIdType2, CEdge*> _edge_map;
  std::map<FaceIdType3, CFace*> _face_map;
};

class MeshGraphBuilder_Hex : public MeshGraphBuilder {
public:
};

#endif
