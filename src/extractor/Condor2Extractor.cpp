#include <assert.h>
#include <list>
#include <set>
#include <libmesh/numeric_vector.h>
#include <libmesh/dof_map.h>
#include "common/Utils.hpp"
#include "Condor2Extractor.h"
#include "InverseInterpolation.h"

using namespace libMesh; 

Condor2VortexExtractor::Condor2VortexExtractor() :
  _verbose(0)
{
}

Condor2VortexExtractor::~Condor2VortexExtractor()
{
}

void Condor2VortexExtractor::SetDataset(const GLDataset* ds)
{
  VortexExtractor::SetDataset(ds);
  _ds = (const Condor2Dataset*)ds;
}
  
void Condor2VortexExtractor::Extract()
{
  _punctured_elems.clear(); 

  const DofMap &dof_map  = _ds->tsys()->get_dof_map();  
  MeshBase::const_element_iterator it = _ds->mesh()->active_local_elements_begin(); 
  const MeshBase::const_element_iterator end = _ds->mesh()->active_local_elements_end(); 
  const NumericVector<Number> *ts = _ds->tsys()->solution.get(); 
 
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
      double u[3] = {(*ts)(u_di[0]), (*ts)(u_di[1]), (*ts)(u_di[2])}, 
             v[3] = {(*ts)(v_di[0]), (*ts)(v_di[1]), (*ts)(v_di[2])};

      Node *nodes[3] = {side->get_node(0), side->get_node(1), side->get_node(2)};
      double X[3][3] = {{nodes[0]->slice(0), nodes[0]->slice(1), nodes[0]->slice(2)}, 
                        {nodes[1]->slice(0), nodes[1]->slice(1), nodes[1]->slice(2)}, 
                        {nodes[2]->slice(0), nodes[2]->slice(1), nodes[2]->slice(2)}};
      double rho[3] = {sqrt(u[0]*u[0]+v[0]*v[0]), sqrt(u[1]*u[1]+v[1]*v[1]), sqrt(u[2]*u[2]+v[2]*v[2])}, 
             phi[3] = {atan2(v[0], u[0]), atan2(v[1], u[1]), atan2(v[2], u[2])}; 

      // check phase shift
      double delta[3];

      if (_gauge) {
        delta[0] = phi[1] - phi[0] + _ds->GaugeTransformation(X[0], X[1]); 
        delta[1] = phi[2] - phi[1] + _ds->GaugeTransformation(X[1], X[2]); 
        delta[2] = phi[0] - phi[2] + _ds->GaugeTransformation(X[2], X[0]); 
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
      phase_shift += _ds->Flux(X); 
      double critera = -phase_shift / (2*M_PI);
      if (fabs(critera)<0.5f) continue; // not punctured
     
      // update bits
      int chirality = critera>0 ? 1 : -1; 

      if (_gauge) {
        phi[1] = phi[0] + delta1[0]; 
        phi[2] = phi[1] + delta1[1];
        u[1] = rho[1] * cos(phi[1]);
        v[1] = rho[1] * sin(phi[1]); 
        u[2] = rho[2] * cos(phi[2]); 
        v[2] = rho[2] * sin(phi[2]);
      }

      double pos[3]; 
      bool succ = find_zero_triangle(u, v, X, pos); 
      if (succ) {
        pelem->AddPuncturedFace(face, chirality, pos);
      } else {
        fprintf(stderr, "WARNING: punctured but singularities not found\n"); 
      }
    }
  
    if (pelem->Punctured()) 
      _punctured_elems[elem->id()] = pelem;
    else 
      delete pelem;
  }
}

