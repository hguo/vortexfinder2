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
  std::vector<const CFace*> contained_faces;
  std::vector<int> contained_faces_chirality;
  std::vector<int> contained_faces_eid; // the edge id in the corresponding face
};

struct CFace {
  // nodes (ordered)
  std::vector<NodeIdType> nodes;

  // edges (ordered)
  std::vector<CEdge*> edges;
  std::vector<int> edges_chirality;

  // neighbor cells, only two, chirality(cell0)=-1, chirality(cell1)=1
  const CCell *contained_cell0, *contained_cell1;
  int contained_cell0_fid, contained_cell1_fid;

  // utils
  bool on_boundary() const {return contained_cell0 == NULL || contained_cell1 == NULL;}
};

struct CCell {
  // nodes (ordered)
  std::vector<NodeIdType> nodes;

  // faces (ordered)
  std::vector<const CFace*> faces;
  std::vector<int> faces_chirality;

  // neighbor cells (ordered)
  std::vector<const CCell*> neighbor_cells;
};

struct MeshGraph {
  std::vector<CEdge*> edges;
  std::vector<CFace*> faces;
  std::vector<CCell*> cells;

  ~MeshGraph();

  void Clear();
  void SerializeToString(std::string &str) const;
  void ParseFromString(const std::string &str);
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
