#include <assert.h>
#include <list>
#include <set>
#include "extractor.h"
#include "utils.h"

VortexExtractor::VortexExtractor(const Parallel::Communicator &comm)
  : ParallelObject(comm), 
    _verbose(0), 
    _mesh(NULL), 
    _exio(NULL), 
    _eqsys(NULL), 
    _tsys(NULL),
    _gauge(false)
{
}

VortexExtractor::~VortexExtractor()
{
  if (_eqsys) delete _eqsys; 
  if (_exio) delete _exio; 
  if (_mesh) delete _mesh; 
}

void VortexExtractor::SetVerbose(int level)
{
  _verbose = level; 
}

void VortexExtractor::SetMagneticField(const double B[3])
{
  memcpy(_B, B, sizeof(double)*3); 
}

void VortexExtractor::SetKex(double Kex)
{
  _Kex = Kex; 
}

void VortexExtractor::SetGaugeTransformation(bool g)
{
  _gauge = g; 
}

void VortexExtractor::LoadData(const std::string& filename)
{
  /// mesh
  _mesh = new Mesh(comm()); 
  _exio = new ExodusII_IO(*_mesh);
  _exio->read(filename);
  _mesh->allow_renumbering(false); 
  _mesh->prepare_for_use();

  if (Verbose())
    _mesh->print_info(); 

  /// equation systems
  _eqsys = new EquationSystems(*_mesh); 
  
  _tsys = &(_eqsys->add_system<NonlinearImplicitSystem>("GLsys"));
  _u_var = _tsys->add_variable("u", FIRST, LAGRANGE);
  _v_var = _tsys->add_variable("v", FIRST, LAGRANGE); 

  _eqsys->init(); 
  
  if (Verbose())
    _eqsys->print_info();
}

void VortexExtractor::LoadTimestep(int timestep)
{
  assert(_exio != NULL); 

  _timestep = timestep;

  if (Verbose())
    fprintf(stderr, "copying nodal solution... timestep=%d\n", timestep); 

  /// copy nodal data
  _exio->copy_nodal_solution(*_tsys, "u", "u", timestep); 
  _exio->copy_nodal_solution(*_tsys, "v", "v", timestep);
  
  if (Verbose())
    fprintf(stderr, "nodal solution copied, timestep=%d\n", timestep); 
}

void VortexExtractor::Extract()
{
  if (Verbose())
    fprintf(stderr, "extracting singularities on mesh faces...\n"); 
 
  _punctured_elems.clear(); 

  const DofMap &dof_map  = _tsys->get_dof_map();  
  MeshBase::const_element_iterator it = _mesh->active_local_elements_begin(); 
  const MeshBase::const_element_iterator end = _mesh->active_local_elements_end(); 
 
  for (; it!=end; it++) {
    const Elem *elem = *it;
    PuncturedElem<> pelem; 
    
    for (int face=0; face<elem->n_sides(); face++) {
      AutoPtr<Elem> side = elem->side(face); 
      
      std::vector<dof_id_type> u_di, v_di; 
      dof_map.dof_indices(side.get(), u_di, _u_var);
      dof_map.dof_indices(side.get(), v_di, _v_var);
     
      // could use solution->get()
      double u[3] = {(*_tsys->solution)(u_di[0]), (*_tsys->solution)(u_di[1]), (*_tsys->solution)(u_di[2])}, 
             v[3] = {(*_tsys->solution)(v_di[0]), (*_tsys->solution)(v_di[1]), (*_tsys->solution)(v_di[2])};

      Node *nodes[3] = {side->get_node(0), side->get_node(1), side->get_node(2)};
      double X0[3] = {nodes[0]->slice(0), nodes[0]->slice(1), nodes[0]->slice(2)}, 
             X1[3] = {nodes[1]->slice(0), nodes[1]->slice(1), nodes[1]->slice(2)}, 
             X2[3] = {nodes[2]->slice(0), nodes[2]->slice(1), nodes[2]->slice(2)};
      double rho[3] = {sqrt(u[0]*u[0]+v[0]*v[0]), sqrt(u[1]*u[1]+v[1]*v[1]), sqrt(u[2]*u[2]+v[2]*v[2])}, 
             phi[3] = {atan2(v[0], u[0]), atan2(v[1], u[1]), atan2(v[2], u[2])}; 

      // check phase shift
      double flux = 0.f; // TODO: need to compute the flux correctly
      double delta[3];

      if (_gauge) {
        delta[0] = phi[1] - phi[0] - gauge_transformation(X0, X1, _Kex, _B); 
        delta[1] = phi[2] - phi[1] - gauge_transformation(X1, X2, _Kex, _B); 
        delta[2] = phi[0] - phi[2] - gauge_transformation(X2, X0, _Kex, _B); 
      } else {
        delta[0] = phi[1] - phi[0]; 
        delta[1] = phi[2] - phi[1]; 
        delta[2] = phi[0] - phi[2]; 
      }

      double delta1[3];  
      double phase_shift = 0.f; 
      for (int k=0; k<3; k++) {
        delta1[k] = mod2pi(delta[k] + M_PI) - M_PI; 
        phase_shift += delta1[k]; 
      }
      phase_shift += flux; 
      double critera = phase_shift / (2*M_PI);
      if (fabs(critera)<0.5f) continue; // not punctured
     
      // update bits
      int chirality = lround(critera);
      pelem.elem_id = elem->id(); 
      pelem.SetChirality(face, chirality);
      pelem.SetPuncturedFace(face);

      if (_gauge) {
        phi[1] = phi[0] + delta1[0]; 
        phi[2] = phi[1] + delta1[1];
        u[1] = rho[1] * cos(phi[1]);
        v[1] = rho[1] * sin(phi[1]); 
        u[2] = rho[2] * cos(phi[2]); 
        v[2] = rho[2] * sin(phi[2]);
      }

      double pos[3]; 
      bool succ = find_zero_triangle(u, v, X0, X1, X2, pos); 
      if (succ) {
        pelem.SetPuncturedPoint(face, pos); 
      } else {
        fprintf(stderr, "WARNING: punctured but singularities not found\n"); 
      }
    }
  
    if (pelem.Valid()) {
      _punctured_elems[elem->id()] = pelem; 
      // fprintf(stderr, "elem_id=%d, bits=%s\n", 
      //     elem->id(), pelem.bits.to_string().c_str()); 
    }
  }
}

void VortexExtractor::Trace()
{
  _vortex_objects.clear();

  while (!_punctured_elems.empty()) {
    /// 1. sort punctured elems into connected ordinary/special ones
    std::list<PuncturedElemMap<>::iterator> to_erase, to_visit;
    to_visit.push_back(_punctured_elems.begin()); 

    PuncturedElemMap<> ordinary_pelems, special_pelems; 
    while (!to_visit.empty()) { // depth-first search
      PuncturedElemMap<>::iterator it = to_visit.front();
      to_visit.pop_front();
      if (it->second.visited) continue; 

      Elem *elem = _mesh->elem(it->first); 
      for (int face=0; face<4; face++) { // for 4 faces, in either directions
        Elem *neighbor = elem->neighbor(face); 
        if (it->second.IsPunctured(face) && neighbor != NULL) {
          PuncturedElemMap<>::iterator it1 = _punctured_elems.find(neighbor->id());
          assert(it1 != _punctured_elems.end());
          if (!it1->second.visited)
            to_visit.push_back(it1); 
        }
      }

      if (it->second.IsSpecial()) 
        special_pelems[it->first] = it->second;
      else 
        ordinary_pelems[it->first] = it->second; 

      it->second.visited = true; 
      to_erase.push_back(it);
    }
   
    for (std::list<PuncturedElemMap<>::iterator>::iterator it = to_erase.begin(); it != to_erase.end(); it ++)
      _punctured_elems.erase(*it);
    to_erase.clear(); 
   
#if 1
    /// 2. trace vortex lines
    VortexObject vortex_object; 
    //// 2.1 special punctured elems
    for (PuncturedElemMap<>::iterator it = special_pelems.begin(); it != special_pelems.end(); it ++) {
      std::list<double> line;
      Elem *elem = _mesh->elem(it->first);
      Point centroid = elem->centroid(); 
      line.push_back(centroid(0)); line.push_back(centroid(1)); line.push_back(centroid(2));
      vortex_object.AddVortexLine(line); 
    }
    if (vortex_object.size() > 0)
      fprintf(stderr, "# of SPECIAL punctured elems: %lu\n", vortex_object.size()); 

    //// 2.2 ordinary punctured elems
    for (PuncturedElemMap<>::iterator it = ordinary_pelems.begin(); it != ordinary_pelems.end(); it ++) 
      it->second.visited = false; 
    while (!ordinary_pelems.empty()) {
      PuncturedElemMap<>::iterator seed = ordinary_pelems.begin(); 
      bool special; 
      std::list<double> line; 
      to_erase.push_back(seed);

      // trace forward (chirality = 1)
      ElemIdType id = seed->first;       
      while (1) {
        int face; 
        double pos[3];
        bool traced = false; 
        Elem *elem = _mesh->elem(id); 
        PuncturedElemMap<>::iterator it = ordinary_pelems.find(id);
        if (it == ordinary_pelems.end()) {
          special = true;
          it = special_pelems.find(id);
        } else {
          special = false;
          // if (it->second.visited) break;  // avoid loop
          // else it->second.visited = true; 
        }
        
        for (face=0; face<4; face++) 
          if (it->second.Chirality(face) == 1) {
            if (it != seed) 
              to_erase.push_back(it); 
            it->second.GetPuncturedPoint(face, pos);
            // fprintf(stderr, "%f, %f, %f\n", pos[0], pos[1], pos[2]); 
            line.push_back(pos[0]); line.push_back(pos[1]); line.push_back(pos[2]);
            Elem *neighbor = elem->neighbor(face);
            if (neighbor != NULL) {
              id = neighbor->id(); 
              if (special)  // `downgrade' the special element
                it->second.RemovePuncturedFace(face); 
              traced = true; 
            }
          }

        if (!traced) break;
      }

      // trace backward (chirality = -1)
      line.pop_front(); line.pop_front(); line.pop_front(); // remove the seed point
      id = seed->first;       
      while (1) {
        int face; 
        double pos[3];
        bool traced = false; 
        Elem *elem = _mesh->elem(id); 
        PuncturedElemMap<>::iterator it = ordinary_pelems.find(id);
        if (it == ordinary_pelems.end()) {
          special = true;
          it = special_pelems.find(id);
        } else {
          special = false;
          // if (it->second.visited) break; // avoid loop
          // else it->second.visited = true; 
        }
        
        for (face=0; face<4; face++) 
          if (it->second.Chirality(face) == -1) {
            if (it != seed) 
              to_erase.push_back(it); 
            it->second.GetPuncturedPoint(face, pos);
            // fprintf(stderr, "%f, %f, %f\n", pos[0], pos[1], pos[2]); 
            line.push_front(pos[2]); line.push_front(pos[1]); line.push_front(pos[0]); 
            Elem *neighbor = elem->neighbor(face);
            if (neighbor != NULL) {
              id = neighbor->id(); 
              if (special)  // `downgrade' the special element
                it->second.RemovePuncturedFace(face); 
              traced = true; 
            }
          }
        if (!traced) break;
      }
      
      for (std::list<PuncturedElemMap<>::iterator>::iterator it = to_erase.begin(); it != to_erase.end(); it ++)
        ordinary_pelems.erase(*it);
      to_erase.clear();

      vortex_object.AddVortexLine(line);

      fprintf(stderr, "#ordinary=%ld\n", ordinary_pelems.size()); 
    }

    _vortex_objects.push_back(vortex_object); 

    fprintf(stderr, "# of lines in vortex_object: %lu\n", vortex_object.size());
    int count = 0; 
    for (VortexObject::iterator it = vortex_object.begin(); it != vortex_object.end(); it ++) {
      fprintf(stderr, " - line %d, # of vertices: %lu\n", count ++, it->size()/3); 
    }
#endif
  }
    
  // fprintf(stderr, "# of vortex objects: %lu\n", vortex_objects.size());
}

void VortexExtractor::WriteVortexObjects(const std::string& filename)
{
  ::WriteVortexObjects(filename, _vortex_objects); 
}
