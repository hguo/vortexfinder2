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

  MeshBase::const_element_iterator it = ds->mesh()->active_local_elements_begin(); 
  const MeshBase::const_element_iterator end = ds->mesh()->active_local_elements_end(); 
 
  for (; it!=end; it++) {
    const Elem *elem = *it;
    ExtractElem(elem->id());
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
  return find_zero_triangle(re, im, X, pos);
}
