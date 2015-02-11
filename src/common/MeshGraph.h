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

EdgeIdType2 AlternateEdge(EdgeIdType2 e2, int chirality);
FaceIdType3 AlternateFace(FaceIdType3 f3, int rotation, int chirality);
FaceIdType4 AlternateFace(FaceIdType4 f4, int rotation, int chirality);

struct CEdge {
  // nodes
  NodeIdType node0, node1;

  // neighbor faces (unordered)
  std::vector<FaceIdType> contained_faces;
  std::vector<int> contained_faces_chirality;
  std::vector<int> contained_faces_eid; // the edge id in the corresponding face

  CEdge() {
    contained_faces.reserve(12);
    contained_faces_chirality.reserve(12);
    contained_faces_eid.reserve(12);
  }
};

struct CFace {
  // nodes (ordered)
  std::vector<NodeIdType> nodes;

  // edges (ordered)
  std::vector<EdgeIdType> edges;
  std::vector<int> edges_chirality;

  // neighbor cells, only two, chirality(cell0)=-1, chirality(cell1)=1
  CellIdType contained_cell0, contained_cell1;
  int contained_cell0_fid, contained_cell1_fid;

  // utils
  // bool on_boundary() const {return contained_cell0 == NULL || contained_cell1 == NULL;}
  CFace() {
    nodes.reserve(3);
    edges.reserve(6);
    edges_chirality.reserve(6);
  }
};

struct CCell {
  // nodes (ordered)
  std::vector<NodeIdType> nodes;

  // faces (ordered)
  std::vector<FaceIdType> faces;
  std::vector<int> faces_chirality;

  // neighbor cells (ordered)
  std::vector<CellIdType> neighbor_cells;

  CCell() {
    nodes.reserve(4);
    faces.reserve(4);
    faces_chirality.reserve(4);
    neighbor_cells.reserve(4);
  }
};

struct MeshGraph {
  std::vector<CEdge> edges;
  std::vector<CFace> faces;
  std::vector<CCell> cells;

  ~MeshGraph();

  void Clear();
  void SerializeToString(std::string &str) const;
  bool ParseFromString(const std::string &str);

  void SerializeToFile(const std::string& filename) const;
  bool ParseFromFile(const std::string& filename);
};


class MeshGraphBuilder {
public:
  explicit MeshGraphBuilder(MeshGraph& mg);
  virtual ~MeshGraphBuilder() {}
 
  // virtual void Build() = 0;

protected:
  MeshGraph &_mg;
};

class MeshGraphBuilder_Tet : public MeshGraphBuilder {
public:
  explicit MeshGraphBuilder_Tet(MeshGraph& mg) : MeshGraphBuilder(mg) {}
  ~MeshGraphBuilder_Tet() {}

  CellIdType AddCell(
      const std::vector<NodeIdType> &nodes, 
      const std::vector<CellIdType> &neighbors, 
      const std::vector<FaceIdType3> &faces);

  // void Build();

private:
  EdgeIdType AddEdge(EdgeIdType2 e2, int &chirality, FaceIdType f, int eid);
  FaceIdType AddFace(FaceIdType3 f3, int &chirality, CellIdType c, int fid);
  EdgeIdType GetEdge(EdgeIdType2 e2, int &chirality);
  FaceIdType GetFace(FaceIdType3 f3, int &chirality);

private:
  std::map<EdgeIdType2, EdgeIdType> _edge_map;
  std::map<FaceIdType3, FaceIdType> _face_map;
};

class MeshGraphBuilder_Hex : public MeshGraphBuilder {
public:
};

#endif
