#include <assert.h>
#include <list>
#include <set>
#include <libmesh/numeric_vector.h>
#include <libmesh/dof_map.h>
#include "common/Utils.hpp"
#include "Condor2Extractor.h"
#include "InverseInterpolation.h"

using namespace libMesh; 

Condor2VortexExtractor::Condor2VortexExtractor()
{
}

Condor2VortexExtractor::~Condor2VortexExtractor()
{
}

void Condor2VortexExtractor::Extract()
{
  _punctured_elems.clear(); 

  const Condor2Dataset *ds = (const Condor2Dataset*)_dataset;

#if 1
  MeshBase::const_element_iterator it = ds->mesh()->active_local_elements_begin(); 
  const MeshBase::const_element_iterator end = ds->mesh()->active_local_elements_end(); 
 
  for (; it!=end; it++) {
    const Elem *elem = *it;
    ExtractElem(elem->id());
  }
#else
  for (MeshWrapper::const_face_iterator it = ds->mesh()->face_begin();
       it != ds->mesh()->face_end(); it ++) 
  {
    const Face *f = it->second;
    ExtractFace(f);
  }
#endif
}

void Condor2VortexExtractor::ExtractFace(const Face* f)
{
  const Condor2Dataset *ds = (const Condor2Dataset*)_dataset;
  const int nnodes = ds->NrNodesPerFace();
  double X[3][3], A[3][3], re[3], im[3];

  ds->GetFaceValues(f, X, A, re, im);
    
  // compute rho & phi
  double rho[nnodes], phi[nnodes];
  for (int i=0; i<nnodes; i++) {
    rho[i] = sqrt(re[i]*re[i] + im[i]*im[i]);
    phi[i] = atan2(im[i], re[i]);
  }

  // calculating phase shift
  double delta[nnodes], phase_shift = 0;
  for (int i=0; i<nnodes; i++) {
    int j = (i+1) % nnodes;
    delta[i] = phi[j] - phi[i]; 
    if (_gauge) 
      delta[i] += _dataset->GaugeTransformation(X[i], X[j], A[i], A[j]) + _dataset->QP(X[i], X[j]); 
    delta[i] = mod2pi(delta[i] + M_PI) - M_PI;
    phase_shift -= delta[i];
    phase_shift -= _dataset->LineIntegral(X[i], X[j], A[i], A[j]);
  }

  // check if punctured
  double critera = phase_shift / (2*M_PI);
  if (fabs(critera)<0.5) return; // not punctured

  // chirality
  int chirality = critera>0 ? 1 : -1;

  // gauge transformation
  if (_gauge) { 
    for (int i=1; i<nnodes; i++) {
      phi[i] = phi[i-1] + delta[i-1];
      re[i] = rho[i] * cos(phi[i]); 
      im[i] = rho[i] * sin(phi[i]);
    }
  }

  // find zero
  double pos[3];
  if (FindZero(X, re, im, pos)) {
    AddPuncturedFace(f->elem_front, f->elem_face_front, chirality, pos);
    AddPuncturedFace(f->elem_back, f->elem_face_back, -chirality, pos);
    // fprintf(stderr, "punctured point p={%f, %f, %f}\n", pos[0], pos[1], pos[2]);
  } else {
    fprintf(stderr, "WARNING: punctured but singularity not found.\n");
  }
}

PuncturedElem* Condor2VortexExtractor::NewPuncturedElem(ElemIdType id) const
{
  PuncturedElem *p = new PuncturedElemTet;
  p->Init();
  p->SetElemId(id);
  return p;
}
  
bool Condor2VortexExtractor::FindZero(const double X[][3], const double re[], const double im[], double pos[3]) const
{
  return find_zero_triangle(re, im, X, pos, 0.05);
}
