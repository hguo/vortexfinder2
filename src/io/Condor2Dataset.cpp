#include <cassert>
#include <cfloat>
#include <libmesh/dof_map.h>
#include "Condor2Dataset.h"
#include "common/DataInfo.pb.h"
#include "common/Utils.hpp"

using namespace libMesh;

Condor2Dataset::Condor2Dataset(const Parallel::Communicator &comm) :
  ParallelObject(comm), 
  _eqsys(NULL), 
  _exio(NULL), 
  _mesh(NULL),
  _locator(NULL)
{
}

Condor2Dataset::~Condor2Dataset()
{
  if (_eqsys) delete _eqsys;
  if (_exio) delete _exio;
  if (_locator) delete _locator;
  if (_mesh) delete _mesh;
}

void Condor2Dataset::PrintInfo(int) const
{
  // TODO
}

void Condor2Dataset::SerializeDataInfoToString(std::string& buf) const
{
  PBDataInfo pb;

  pb.set_model(PBDataInfo::CONDOR2);
  pb.set_name(_data_name);

  if (Lengths()[0]>0) {
    pb.set_ox(Origins()[0]); 
    pb.set_oy(Origins()[1]); 
    pb.set_oz(Origins()[2]); 
    pb.set_lx(Lengths()[0]); 
    pb.set_ly(Lengths()[1]); 
    pb.set_lz(Lengths()[2]); 
  }

  pb.set_kex(Kex());

  pb.SerializeToString(&buf);
}

bool Condor2Dataset::OpenDataFile(const std::string& filename)
{
  _data_name = filename;

  /// mesh
  _mesh = new Mesh(comm()); 
  _exio = new ExodusII_IO(*_mesh);
  _exio->read(filename);
  _mesh->allow_renumbering(false); 
  _mesh->prepare_for_use();

  /// mesh graph
  BuildMeshGraph();

  /// equation systems
  _eqsys = new EquationSystems(*_mesh); 
  
  _tsys = &(_eqsys->add_system<NonlinearImplicitSystem>("GLsys"));
  _rho_var = _tsys->add_variable("rho", FIRST, LAGRANGE);
  _phi_var = _tsys->add_variable("phi", FIRST, LAGRANGE); 
  
  _asys = &(_eqsys->add_system<System>("Auxsys"));
  _Ax_var = _asys->add_variable("Ax", FIRST, LAGRANGE);
  _Ay_var = _asys->add_variable("Ay", FIRST, LAGRANGE); 
  _Az_var = _asys->add_variable("Az", FIRST, LAGRANGE);
  // _rho_var = _asys->add_variable("rho", FIRST, LAGRANGE);
  // _phi_var = _asys->add_variable("phi", FIRST, LAGRANGE);

  _eqsys->init(); 

  /// point locator
  _locator = new PointLocatorTree(*_mesh);

  // it takes some time (~0.5s) to compute the bounding box. is there any better way to get this information?
  ProbeBoundingBox();

  _valid = true;
  return true; 
}

void Condor2Dataset::CloseDataFile()
{
  if (_eqsys) delete _eqsys; 
  if (_exio) delete _exio; 
  if (_mesh) delete _mesh; 
}

void Condor2Dataset::BuildMeshGraph()
{
  fprintf(stderr, "initializing mesh graph..\n");
  if (LoadDefaultMeshGraph()) {
    fprintf(stderr, "loaded default mesh graph:\n");
    fprintf(stderr, "#node=%u, #edge=%u, #face=%u, #cell=%u\n", 
        _mesh->n_nodes(), _mg->NEdges(), _mg->NFaces(), _mg->NCells());
    return;
  }

  _mg = new class MeshGraph;
  MeshGraphBuilder_Tet *builder = new MeshGraphBuilder_Tet(mesh()->n_elem(), *_mg);
  
  MeshBase::const_element_iterator it = mesh()->local_elements_begin(); 
  const MeshBase::const_element_iterator end = mesh()->local_elements_end(); 

  for (; it!=end; it++) {
    const Elem *e = *it;
    std::vector<NodeIdType> nodes;
    std::vector<CellIdType> neighbors;
    std::vector<FaceIdType3> faces;

    for (int i=0; i<e->n_nodes(); i++)
      nodes.push_back(e->node(i));

    for (int i=0; i<e->n_sides(); i++) {
      if (e->neighbor(i) != NULL) 
        neighbors.push_back(e->neighbor(i)->id());
      else 
        neighbors.push_back(UINT_MAX);

      AutoPtr<Elem> f = e->side(i);
      faces.push_back(std::make_tuple(f->node(0), f->node(1), f->node(2)));
    }

    builder->AddCell(e->id(), nodes, neighbors, faces);
  }

  delete builder;
  
  fprintf(stderr, "mesh graph built..\n");
  fprintf(stderr, "#node=%u, #edge=%u, #face=%u, #cell=%u\n", 
      _mesh->n_nodes(), _mg->NEdges(), _mg->NFaces(), _mg->NCells());

  SaveDefaultMeshGraph();
  fprintf(stderr, "mesh graph saved.\n");
}

void Condor2Dataset::ProbeBoundingBox()
{
  float L[3] = {DBL_MAX, DBL_MAX, DBL_MAX}, 
         U[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};

  MeshBase::const_node_iterator it = mesh()->local_nodes_begin(); 
  const MeshBase::const_node_iterator end = mesh()->local_nodes_end();

  for (; it != end; it++) {
    L[0] = std::min(L[0], (*it)->slice(0));
    L[1] = std::min(L[1], (*it)->slice(1));
    L[2] = std::min(L[2], (*it)->slice(2));
    U[0] = std::max(U[0], (*it)->slice(0));
    U[1] = std::max(U[1], (*it)->slice(1));
    U[2] = std::max(U[2], (*it)->slice(2));
  }

  _h[0].origins[0] = L[0]; 
  _h[0].origins[1] = L[1]; 
  _h[0].origins[2] = L[2]; 
  _h[0].lengths[0] = U[0] - L[0]; 
  _h[0].lengths[1] = U[1] - L[1]; 
  _h[0].lengths[2] = U[2] - L[2];
}

void Condor2Dataset::LoadTimeStep_(int timestep)
{
  assert(_exio != NULL); 

  fprintf(stderr, "copying nodal solution, timestep=%d\n", timestep);

  _exio->copy_nodal_solution(*_tsys, "u", "u", timestep); 
  _exio->copy_nodal_solution(*_tsys, "v", "v", timestep);
  _exio->copy_nodal_solution(*_asys, "Ax", "A_x", timestep);
  _exio->copy_nodal_solution(*_asys, "Ay", "A_y", timestep);
  _exio->copy_nodal_solution(*_asys, "Az", "A_z", timestep);
}

void Condor2Dataset::LoadTimeStep(int timestep, int slot)
{
  SetTimeStep(timestep, slot);
  LoadTimeStep_(timestep);

  if (slot == 0) {
    _ts = _tsys->solution->clone();
    _as = _asys->solution->clone();
  } else {
    if (_ts1.get() != NULL) {
      _ts = _ts1;
      _as = _as1;
    }

    _ts1 = _tsys->solution->clone();
    _as1 = _asys->solution->clone();

    GLDatasetBase::RotateTimeSteps();
  }

  return true; // FIXME
}

#if 0
std::vector<ElemIdType> Condor2Dataset::GetNeighborIds(ElemIdType elem_id) const
{
  std::vector<ElemIdType> neighbors(4);
  const Elem* elem = mesh()->elem(elem_id); 

  for (int face=0; face<4; face++) {
    const Elem* elem1 = elem->neighbor(face);
    if (elem1 != NULL)
      neighbors[face] = elem1->id();
    else 
      neighbors[face] = UINT_MAX;
  }

  return neighbors;
}
#endif

CellIdType Condor2Dataset::Pos2CellId(const float X[]) const
{
  Point p(X[0], X[1], X[2]);
  
  const Elem *e = (*_locator)(p);
  if (e == NULL) return UINT_MAX;
  else return e->id();
}

bool Condor2Dataset::A(const float X[3], float A[3], int slot) const
{
  Point p(X[0], X[1], X[2]);

  const Elem *e = (*_locator)(p);
  if (e == NULL) return false;

  A[0] = asys()->point_value(_Ax_var, p, e);
  A[1] = asys()->point_value(_Ay_var, p, e);
  A[2] = asys()->point_value(_Az_var, p, e);

  return true;
}

bool Condor2Dataset::A(NodeIdType, float A[3], int slot) const
{
  // TODO
  return false;
}

bool Condor2Dataset::Pos(NodeIdType, float X[3]) const
{
  // TODO
  return false;
}

#if 0
bool Condor2Dataset::Psi(const float X[3], float &re, float &im, int slot) const
{
  if (X[0] < Origins()[0] || X[0] > Origins()[0] + Lengths()[0] ||
      X[1] < Origins()[1] || X[1] > Origins()[1] + Lengths()[1] || 
      X[2] < Origins()[2] || X[2] > Origins()[2] + Lengths()[2])
    return false;
  
  Point p(X[0], X[1], X[2]);
  const Elem *e = (*_locator)(p);
  if (e == NULL) return false;

  re = tsys()->point_value(_u_var, p, e);
  im = tsys()->point_value(_v_var, p, e);

  return true;
}
#endif

bool Condor2Dataset::Supercurrent(NodeIdType, float J[3], int slot) const
{
  // TODO
  return false;
}

bool Condor2Dataset::Supercurrent(const float X[3], float J[3], int slot) const
{
  return false;
#if 0
  Point p(X[0], X[1], X[2]);

  const Elem *e = (*_locator)(p);
  if (e == NULL) return false;

  NumberVectorValue A;
  A(0) = asys()->point_value(_Ax_var, p, e);
  A(1) = asys()->point_value(_Ay_var, p, e);
  A(2) = asys()->point_value(_Az_var, p, e);

  float rho, phi;
  Gradient du, dv;

  rho = tsys()->point_value(_u_var, p, e);
  phi = tsys()->point_value(_v_var, p, e);
  du = tsys()->point_gradient(_u_var, p, e);
  dv = tsys()->point_gradient(_v_var, p, e);

  float amp = sqrt(u*u + v*v);

  NumberVectorValue Js = (u*dv - v*du)/amp - A;

  J[0] = Js(0);
  J[1] = Js(1);
  J[2] = Js(2);

  return true;
#endif
}
 
const Elem* Condor2Dataset::LocateElemCoherently(const float X[3]) const
{
  static const Elem* e_last = NULL;
  
  if (X[0] < Origins()[0] || X[0] > Origins()[0] + Lengths()[0] ||
      X[1] < Origins()[1] || X[1] > Origins()[1] + Lengths()[1] || 
      X[2] < Origins()[2] || X[2] > Origins()[2] + Lengths()[2])
    return NULL;
  
  Point p(X[0], X[1], X[2]);

  /// check if p is inside last queried element or neighbors
  if (e_last != NULL) {
    if (e_last->contains_point(p)) {
      // fprintf(stderr, "hit last.\n");
      return e_last;
    }
    else {
      for (int i=0; i<e_last->n_neighbors(); i++) {
        const Elem* e = e_last->neighbor(i);
        if (e != NULL && e->contains_point(p)) {
          // fprintf(stderr, "hit last neighbor.\n");
          e_last = e;
          return e;
        }
      }
    }
  }
  
  e_last = (*_locator)(p);
  return e_last;
}

bool Condor2Dataset::OnBoundary(ElemIdType id) const
{
  const Elem* elem = mesh()->elem(id);
  if (elem == NULL) return false;
  else return elem->on_boundary();
}

#if 0
bool Condor2Dataset::GetFaceValues(const Face *f, float X[][3], float A[][3], float re[], float im[]) const
{
  const NumericVector<Number> &ts = *_ts;
  const NumericVector<Number> &as = *_as;

  for (int i=0; i<3; i++) {
    const Node& node = _mesh->node(f->nodes[i]);

    for (int j=0; j<3; j++) 
      X[i][j] = node(j);

    A[i][0] = as( node.dof_number(asys()->number(), _Ax_var, 0) );
    A[i][1] = as( node.dof_number(asys()->number(), _Ay_var, 0) );
    A[i][2] = as( node.dof_number(asys()->number(), _Az_var, 0) );
    re[i] = ts( node.dof_number(tsys()->number(), _u_var, 0) );
    im[i] = ts( node.dof_number(tsys()->number(), _v_var, 0) );
  }

  return true;
}

bool Condor2Dataset::GetFacePrismValues(const Face* f, float X[][3],
    float A[6][3], float re[6], float im[6]) const
{
  const NumericVector<Number> &ts = *_ts;
  const NumericVector<Number> &ts1 = *_ts1;
  const NumericVector<Number> &as = *_as;
  const NumericVector<Number> &as1 = *_as1;

  for (int i=0; i<3; i++) {
    const Node& node = _mesh->node(f->nodes[i]);

    for (int j=0; j<3; j++) 
      X[i][j] = node(j);

    A[i][0] = as( node.dof_number(asys()->number(), _Ax_var, 0) );
    A[i][1] = as( node.dof_number(asys()->number(), _Ay_var, 0) );
    A[i][2] = as( node.dof_number(asys()->number(), _Az_var, 0) );
    re[i] = ts( node.dof_number(tsys()->number(), _u_var, 0) );
    im[i] = ts( node.dof_number(tsys()->number(), _v_var, 0) );
    
    A[i+3][0] = as1( node.dof_number(asys()->number(), _Ax_var, 0) );
    A[i+3][1] = as1( node.dof_number(asys()->number(), _Ay_var, 0) );
    A[i+3][2] = as1( node.dof_number(asys()->number(), _Az_var, 0) );
    re[i+3] = ts1( node.dof_number(tsys()->number(), _u_var, 0) );
    im[i+3] = ts1( node.dof_number(tsys()->number(), _v_var, 0) );
  }

  return true;
}
 
bool Condor2Dataset::GetSpaceTimeEdgeValues(const Edge* e, float X[][3], float A[][3], float re[], float im[]) const
{
  const NumericVector<Number> &ts = *_ts;
  const NumericVector<Number> &ts1 = *_ts1;
  const NumericVector<Number> &as = *_as;
  const NumericVector<Number> &as1 = *_as1;
  
  const Node& node0 = _mesh->node(e->node0), 
              node1 = _mesh->node(e->node1);

  for (int j=0; j<3; j++) {
    X[0][j] = node0(j);
    X[1][j] = node1(j);
  }

  A[0][0] = as( node0.dof_number(asys()->number(), _Ax_var, 0) );
  A[0][1] = as( node0.dof_number(asys()->number(), _Ay_var, 0) );
  A[0][2] = as( node0.dof_number(asys()->number(), _Az_var, 0) );

  A[1][0] = as( node1.dof_number(asys()->number(), _Ax_var, 0) );
  A[1][1] = as( node1.dof_number(asys()->number(), _Ay_var, 0) );
  A[1][2] = as( node1.dof_number(asys()->number(), _Az_var, 0) );
  
  A[2][0] = as1( node1.dof_number(asys()->number(), _Ax_var, 0) );
  A[2][1] = as1( node1.dof_number(asys()->number(), _Ay_var, 0) );
  A[2][2] = as1( node1.dof_number(asys()->number(), _Az_var, 0) );
  
  A[3][0] = as1( node0.dof_number(asys()->number(), _Ax_var, 0) );
  A[3][1] = as1( node0.dof_number(asys()->number(), _Ay_var, 0) );
  A[3][2] = as1( node0.dof_number(asys()->number(), _Az_var, 0) );

  re[0] = ts( node0.dof_number(tsys()->number(), _u_var, 0) );
  re[1] = ts( node1.dof_number(tsys()->number(), _u_var, 0) );
  re[2] = ts1( node1.dof_number(tsys()->number(), _u_var, 0) );
  re[3] = ts1( node0.dof_number(tsys()->number(), _u_var, 0) );

  im[0] = ts( node0.dof_number(tsys()->number(), _v_var, 0) );
  im[1] = ts( node1.dof_number(tsys()->number(), _v_var, 0) );
  im[2] = ts1( node1.dof_number(tsys()->number(), _v_var, 0) );
  im[3] = ts1( node0.dof_number(tsys()->number(), _v_var, 0) );

  return true;
}
#endif

void Condor2Dataset::GetFaceValues(const CFace& f, int time, float X[][3], float A[][3], float rho[], float phi[]) const
{
  const NumericVector<Number> &ts = time == 0 ? *_ts : *_ts1;
  const NumericVector<Number> &as = time == 0 ? *_as : *_as1;
 
  const Node nodes[3] = {
    _mesh->node(f.nodes[0]), 
    _mesh->node(f.nodes[1]), 
    _mesh->node(f.nodes[2])};

  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      X[i][j] = nodes[i](j); 
    }

    A[i][0] = as( nodes[i].dof_number(asys()->number(), _Ax_var, 0) );
    A[i][1] = as( nodes[i].dof_number(asys()->number(), _Ay_var, 0) );
    A[i][2] = as( nodes[i].dof_number(asys()->number(), _Az_var, 0) );

    rho[i] = ts( nodes[i].dof_number(tsys()->number(), _rho_var, 0) );
    phi[i] = ts( nodes[i].dof_number(tsys()->number(), _phi_var, 0) );
  }
}

void Condor2Dataset::GetSpaceTimeEdgeValues(const CEdge& e, float X[][3], float A[][3], float rho[], float phi[]) const
{
  const NumericVector<Number> &ts = *_ts;
  const NumericVector<Number> &ts1 = *_ts1;
  const NumericVector<Number> &as = *_as;
  const NumericVector<Number> &as1 = *_as1;
  
  const Node& node0 = _mesh->node(e.node0), 
              node1 = _mesh->node(e.node1);
 
  for (int j=0; j<3; j++) {
    X[0][j] = node0(j);
    X[1][j] = node1(j);
  }

  A[0][0] = as( node0.dof_number(asys()->number(), _Ax_var, 0) );
  A[0][1] = as( node0.dof_number(asys()->number(), _Ay_var, 0) );
  A[0][2] = as( node0.dof_number(asys()->number(), _Az_var, 0) );

  A[1][0] = as( node1.dof_number(asys()->number(), _Ax_var, 0) );
  A[1][1] = as( node1.dof_number(asys()->number(), _Ay_var, 0) );
  A[1][2] = as( node1.dof_number(asys()->number(), _Az_var, 0) );
  
  A[2][0] = as1( node1.dof_number(asys()->number(), _Ax_var, 0) );
  A[2][1] = as1( node1.dof_number(asys()->number(), _Ay_var, 0) );
  A[2][2] = as1( node1.dof_number(asys()->number(), _Az_var, 0) );
  
  A[3][0] = as1( node0.dof_number(asys()->number(), _Ax_var, 0) );
  A[3][1] = as1( node0.dof_number(asys()->number(), _Ay_var, 0) );
  A[3][2] = as1( node0.dof_number(asys()->number(), _Az_var, 0) );

  rho[0] = ts( node0.dof_number(tsys()->number(), _rho_var, 0) );
  rho[1] = ts( node1.dof_number(tsys()->number(), _rho_var, 0) );
  rho[2] = ts1( node1.dof_number(tsys()->number(), _rho_var, 0) );
  rho[3] = ts1( node0.dof_number(tsys()->number(), _rho_var, 0) );

  phi[0] = ts( node0.dof_number(tsys()->number(), _phi_var, 0) );
  phi[1] = ts( node1.dof_number(tsys()->number(), _phi_var, 0) );
  phi[2] = ts1( node1.dof_number(tsys()->number(), _phi_var, 0) );
  phi[3] = ts1( node0.dof_number(tsys()->number(), _phi_var, 0) );
}

float Condor2Dataset::Rho(NodeIdType, int) const
{
  // TODO
  assert(false);
}

float Condor2Dataset::Phi(NodeIdType, int) const
{
  // TODO
  assert(false);
}
