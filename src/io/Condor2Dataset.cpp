#include <cassert>
#include <cfloat>
#include <libmesh/dof_map.h>
#include <libmesh/numeric_vector.h>
#include "Condor2Dataset.h"
#include "common/DataInfo.pb.h"

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

void Condor2Dataset::PrintInfo() const
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

  pb.set_bx(Bx());
  pb.set_by(By());
  pb.set_bz(Bz());

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

  /// equation systems
  _eqsys = new EquationSystems(*_mesh); 
  
  _tsys = &(_eqsys->add_system<NonlinearImplicitSystem>("GLsys"));
  _u_var = _tsys->add_variable("u", FIRST, LAGRANGE);
  _v_var = _tsys->add_variable("v", FIRST, LAGRANGE); 

  _eqsys->init(); 

  /// FE
#if 0
  const DofMap& dof_map = _tsys->get_dof_map();
  FEType fe_type = dof_map.variable_type(2/*vnum*/);
  AutoPtr<FEBase> febase(FEBase::build(_mesh->mesh_dimension(), fe_type));
  _fe = febase;
#endif

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

void Condor2Dataset::ProbeBoundingBox()
{
  double L[3] = {DBL_MAX, DBL_MAX, DBL_MAX}, 
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

  _origins[0] = L[0]; 
  _origins[1] = L[0]; 
  _origins[2] = L[0]; 
  _lengths[0] = U[0] - L[0]; 
  _lengths[1] = U[1] - L[1]; 
  _lengths[2] = U[2] - L[2]; 
}

void Condor2Dataset::LoadTimeStep(int timestep)
{
  assert(_exio != NULL); 

  _timestep = timestep;

  // fprintf(stderr, "copying nodal solution... timestep=%d\n", timestep); 

  /// copy nodal data
  _exio->copy_nodal_solution(*_tsys, "u", "u", timestep); 
  _exio->copy_nodal_solution(*_tsys, "v", "v", timestep);
  
  // fprintf(stderr, "nodal solution copied, timestep=%d\n", timestep); 
}

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

ElemIdType Condor2Dataset::Pos2ElemId(const double X[]) const
{
  Point p(X[0], X[1], X[2]);
  const Elem *elem = (*_locator)(p);

  if (elem == NULL)
    return UINT_MAX;
  else 
    return elem->id();
}

bool Condor2Dataset::Psi(const double X[3], double &re, double &im) const
{
  // TODO
  return false;
}

bool Condor2Dataset::Supercurrent(const double X[3], double J[3]) const
{
  ElemIdType elem_id = Pos2ElemId(X);
  if (elem_id == UINT_MAX) return false;

  const Elem* elem = _mesh->elem(elem_id);
  const DofMap& dof_map = tsys()->get_dof_map();
  const NumericVector<Number> *ts = tsys()->solution.get();

  std::vector<dof_id_type> u_di, v_di;
  dof_map.dof_indices(elem, u_di, u_var());
  dof_map.dof_indices(elem, v_di, v_var());

  double P[4][4];
  for (int i=0; i<4; i++) {
    const Node *node = elem->get_node(i);
    for (int j=0; j<3; j++)
      P[i][j] = node->slice(j);
  }

  double phi[4]; 
  for (int i=0; i<4; i++) {
    double u = (*ts)(u_di[i]), 
           v = (*ts)(v_di[i]); 
    phi[i] = atan2(v, u);
  }

  // TODO: gradient estimation

  return false;
}
  
bool Condor2Dataset::GetFace(ElemIdType id, int face, double X[][3], double re[], double im[]) const
{
  if (id == UINT_MAX) return false;
  
  const Elem* elem = _mesh->elem(id);
  if (elem == NULL) return false;

  AutoPtr<Elem> side = elem->side(face); // TODO: check if side exists
  const DofMap& dof_map = tsys()->get_dof_map();
  const NumericVector<Number> *ts = tsys()->solution.get();

  std::vector<dof_id_type> u_di, v_di; 
  dof_map.dof_indices(side.get(), u_di, u_var());
  dof_map.dof_indices(side.get(), v_di, v_var());

  // coordinates
  for (int i=0; i<3; i++) 
    for (int j=0; j<3; j++) 
      X[i][j] = side->get_node(i)->slice(j);

  // nodal values
  for (int i=0; i<3; i++) {
    re[i] = (*ts)(u_di[i]); 
    im[i] = (*ts)(v_di[i]);
  }

  return true;
}
