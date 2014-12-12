#include <assert.h>
#include <list>
#include <set>
#include "Condor2Extractor.h"
#include "Utils.h"

Condor2VortexExtractor::Condor2VortexExtractor() :
  _verbose(0)
{
}

Condor2VortexExtractor::~Condor2VortexExtractor()
{
}

void Condor2VortexExtractor::SetDataset(const GLDataset* ds)
{
  _ds = (const Condor2Dataset*)ds;
}
  
std::vector<unsigned int> Condor2VortexExtractor::Neighbors(unsigned int elem_id) const
{
  std::vector<unsigned int> neighbors(4);
  const Elem* elem = _ds->mesh()->elem(elem_id); 

  for (int face=0; face<4; face++) {
    const Elem* elem1 = elem->neighbor(face);
    if (elem1 != NULL)
      neighbors[face] = elem1->id();
    else 
      neighbors[face] = UINT_MAX;
  }

  return neighbors;
}

void Condor2VortexExtractor::Extract()
{
  _punctured_elems.clear(); 

  const DofMap &dof_map  = _ds->tsys()->get_dof_map();  
  MeshBase::const_element_iterator it = _ds->mesh()->active_local_elements_begin(); 
  const MeshBase::const_element_iterator end = _ds->mesh()->active_local_elements_end(); 
  const NumericVector<Number> *tsolution = _ds->tsys()->solution.get(); 
 
  for (; it!=end; it++) {
    const Elem *elem = *it;
    PuncturedElemTet *pelem = new PuncturedElemTet;
    pelem->Init();
    pelem->SetElemId(elem->id()); 
    
    for (int face=0; face<elem->n_sides(); face++) {
      AutoPtr<Elem> side = elem->side(face); 
      
      std::vector<dof_id_type> u_di, v_di; 
      dof_map.dof_indices(side.get(), u_di, _ds->u_var());
      dof_map.dof_indices(side.get(), v_di, _ds->v_var());
     
      // could use solution->get()
      double u[3] = {(*tsolution)(u_di[0]), (*tsolution)(u_di[1]), (*tsolution)(u_di[2])}, 
             v[3] = {(*tsolution)(v_di[0]), (*tsolution)(v_di[1]), (*tsolution)(v_di[2])};

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
        delta[0] = phi[1] - phi[0] - gauge_transformation(X0, X1, _ds->Kex(), _ds->B()); 
        delta[1] = phi[2] - phi[1] - gauge_transformation(X1, X2, _ds->Kex(), _ds->B()); 
        delta[2] = phi[0] - phi[2] - gauge_transformation(X2, X0, _ds->Kex(), _ds->B()); 
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
        pelem->AddPuncturedFace(face, chirality, pos);
      } else {
        fprintf(stderr, "WARNING: punctured but singularities not found\n"); 
      }
    }
  
    if (pelem->Punctured()) {
      _punctured_elems[elem->id()] = pelem; 
      // fprintf(stderr, "elem_id=%d, bits=%s\n", 
      //     elem->id(), pelem.bits.to_string().c_str()); 
    }
  }
}

