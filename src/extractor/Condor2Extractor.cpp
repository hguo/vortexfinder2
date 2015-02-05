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

static int n_invalid = 0, 
           n_pure = 0, 
           n_self = 0;

void Condor2VortexExtractor::Extract()
{
  _punctured_elems.clear(); 

  const Condor2Dataset *ds = (const Condor2Dataset*)_dataset;

#if 0
  MeshBase::const_element_iterator it = ds->mesh()->active_local_elements_begin(); 
  const MeshBase::const_element_iterator end = ds->mesh()->active_local_elements_end(); 
 
  for (; it!=end; it++) {
    const Elem *elem = *it;
    ExtractElem(elem->id());
  }
#else
  for (FaceIdType i=0; i<ds->mesh()->NrFaces(); i++) {
    const Face *f = ds->mesh()->GetFace(i);
    // ExtractFace(f);
    ExtractFacePrism(f);
  }
  
  fprintf(stderr, "n_invalid=%d, n_pure=%d, n_self=%d\n", n_invalid, n_pure, n_self);
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

void Condor2VortexExtractor::ExtractFacePrism(const Face* f)
{
  const Condor2Dataset *ds = (const Condor2Dataset*)_dataset;
  const int nnodes = ds->NrNodesPerFace();
  double X[3][3];
  double A[6][3], re[6], im[6];

  ds->GetFacePrismValues(f, X, A, re, im);
 
  // edge 0
  double X0[2][3] = {{X[0][0], X[0][1], X[0][2]}, 
                     {X[1][0], X[1][1], X[1][2]}}, 
         A0[4][3] = {{A[0][0], A[0][1], A[0][2]},
                     {A[1][0], A[1][1], A[1][2]}, 
                     {A[4][0], A[4][1], A[4][2]}, 
                     {A[3][0], A[3][1], A[3][2]}}, 
         re0[4] = {re[0], re[1], re[4], re[3]}, 
         im0[4] = {im[0], im[1], im[4], im[3]};
  
  // edge 1
  double X1[2][3] = {{X[1][0], X[1][1], X[1][2]}, 
                     {X[2][0], X[2][1], X[2][2]}}, 
         A1[4][3] = {{A[1][0], A[1][1], A[1][2]},
                     {A[2][0], A[2][1], A[2][2]}, 
                     {A[5][0], A[5][1], A[5][2]}, 
                     {A[4][0], A[4][1], A[4][2]}}, 
         re1[4] = {re[1], re[2], re[5], re[4]}, 
         im1[4] = {im[1], im[2], im[5], im[4]};
  
  // edge 2
  double X2[2][3] = {{X[2][0], X[2][1], X[2][2]}, 
                     {X[0][0], X[0][1], X[0][2]}}, 
         A2[4][3] = {{A[2][0], A[2][1], A[2][2]},
                    {A[0][0], A[0][1], A[0][2]}, 
                    {A[3][0], A[3][1], A[3][2]}, 
                    {A[5][0], A[5][1], A[5][2]}}, 
         re2[4] = {re[2], re[0], re[3], re[5]}, 
         im2[4] = {im[2], im[0], im[3], im[5]};
  
  int fp0 = -CheckFace(X, A, re, im), 
      fp1 = CheckFace(X+3, A+3, re+3, im+3);
  int vfp[3] = {
    CheckVirtualFace(X0, A0, re0, im0),
    CheckVirtualFace(X1, A1, re1, im1),
    CheckVirtualFace(X2, A2, re2, im2)};

  bool punctured = fp0 || fp1 || vfp[0] || vfp[1] || vfp[2]; 
  bool pure = punctured && !fp0 && !fp1;
  bool self = fp0 && fp1;
  int fpsum = fp0 + fp1 + vfp[0] + vfp[1] + vfp[2];

  if (fpsum != 0) n_invalid ++; 
  if (pure) n_pure ++; 
  if (self) n_self ++;

  if (punctured) 
    fprintf(stderr, "%d\t%d\t%d\t%d\t%d\n", fp0, fp1, vfp[0], vfp[1], vfp[2]);
}

int Condor2VortexExtractor::CheckFace(double X[3][3], double A[3][3], double re[3], double im[3]) const
{
  double rho[3], phi[3];
  for (int i=0; i<3; i++) {
    rho[i] = sqrt(re[i]*re[i] + im[i]*im[i]);
    phi[i] = atan2(im[i], re[i]);
  }
  
  double delta[3], phase_shift = 0;
  for (int i=0; i<3; i++) {
    int j = (i+1) % 3;
    delta[i] = phi[j] - phi[i]; 
    if (_gauge) 
      delta[i] += _dataset->GaugeTransformation(X[i], X[j], A[i], A[j]) + _dataset->QP(X[i], X[j]); 
    delta[i] = mod2pi(delta[i] + M_PI) - M_PI;
    phase_shift -= delta[i];
    phase_shift -= _dataset->LineIntegral(X[i], X[j], A[i], A[j]);
  }

  double critera = phase_shift / (2*M_PI);
  if (critera > 0.5) return 1; 
  else if (critera < -0.5) return -1;
  else return 0;
}

int Condor2VortexExtractor::CheckVirtualFace(double X[2][3], double A[4][3], double re[4], double im[4]) const
{
  double rho[4], phi[4];
  for (int i=0; i<4; i++) {
    rho[i] = sqrt(re[i]*re[i] + im[i]*im[i]);
    phi[i] = atan2(im[i], re[i]);
  }

  double delta[4] = {
    phi[1] - phi[0] + _dataset->GaugeTransformation(X[0], X[1], A[0], A[1]), // TODO: QP
    phi[2] - phi[1], // no gauge transformation along the time axis
    phi[3] - phi[2] + _dataset->GaugeTransformation(X[1], X[0], A[3], A[2]), 
    phi[0] - phi[3]
  };

  for (int i=0; i<4; i++) 
    delta[i] = mod2pi(delta[i] + M_PI) - M_PI;

  // TODO: is it necessary to do line integral? 
  
  double phase_shift = -(delta[0] + delta[1] + delta[2] + delta[3]);
  double critera = phase_shift / (2*M_PI);

#if 1
  if (fabs(critera)<0.5) return 0;

  // gauge transformation
  for (int i=1; i<4; i++) {
    phi[i] = phi[i-1] + delta[i-1];
    re[i] = rho[i] * cos(phi[i]); 
    im[i] = rho[i] * sin(phi[i]);
  }

  // find zero
  double p[2];
  bool succ = find_zero_unit_quad_bilinear(re, im, p);
  if (succ) {
    double pos[3] = {
      (1-p[0]) * X[0][0] + p[0] * X[1][0], 
      (1-p[0]) * X[0][1] + p[0] * X[1][1], 
      (1-p[0]) * X[0][2] + p[0] * X[1][2]}; 
    fprintf(stderr, "X={%f, %f, %f}, t=%f\n", pos[0], pos[1], pos[2], p[1]);
  } else {
    fprintf(stderr, "zero not found.\n");
  }
#endif

  if (critera > 0.5) return 1; 
  else if (critera < -0.5) return -1;
  else return 0;
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
