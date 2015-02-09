#include "Extractor.h"
#include "common/Utils.hpp"
#include "io/GLDataset.h"
#include <set>
#include <cassert>

VortexExtractor::VortexExtractor() :
  _dataset(NULL), 
  _gauge(false)
{

}

VortexExtractor::~VortexExtractor()
{

}

void VortexExtractor::SetDataset(const GLDataset* ds)
{
  _dataset = ds;
}

void VortexExtractor::SetGaugeTransformation(bool g)
{
  _gauge = g; 
}

void VortexExtractor::WriteVortexObjects(const std::string& filename)
{
  ::WriteVortexObjects(filename, _vortex_objects); 
}

void VortexExtractor::AddPuncturedFace(FaceIdType id, int time, int chirality, const double pos[])
{
  // face
  PuncturedFace pf;

  pf.chirality = chirality;
  memcpy(pf.pos, pos, sizeof(double)*3);

  if (time == 0)
    _punctured_faces[id] = pf;
  else 
    _punctured_faces1[id] = pf;
}

void VortexExtractor::AddPuncturedEdge(EdgeIdType id, int chirality, double t)
{
  // edge
  PuncturedEdge pe;

  pe.chirality = chirality;
  pe.t = t;

  _punctured_edges[id] = pe;
}
  
bool VortexExtractor::FindSpaceTimeEdgeZero(const double re[], const double im[], double &t) const
{
  double p[2];
  if (!find_zero_unit_quad_bilinear(re, im, p))
    return false;

  t = p[1];
  return true;
}

void VortexExtractor::Trace()
{
  fprintf(stderr, "Tracing over time, #pf0=%ld, #pf1=%ld, #pe=%ld\n", 
      _punctured_faces.size(), _punctured_faces1.size(), _punctured_edges.size());
  const MeshGraph &mg = _dataset->MeshGraph();

  for (std::map<FaceIdType, PuncturedFace>::iterator it = _punctured_faces.begin(); 
       it != _punctured_faces.end(); it ++) 
  {
    fprintf(stderr, "fid=%u:\n", it->first);

    std::vector<FaceIdType> related;
    
    std::list<FaceIdType> to_visit;
    std::list<int> to_visit_chirality; // face chirality
    std::list<double >to_visit_time;
    std::set<FaceIdType> visited;

    to_visit.push_back(it->first);
    to_visit_chirality.push_back(it->second.chirality);
    to_visit_time.push_back(0);

    while (!to_visit.empty()) {
      FaceIdType current = to_visit.front();
      int current_chirality = to_visit_chirality.front();
      double current_time = to_visit_time.front();

      to_visit.pop_front();
      to_visit_chirality.pop_front();
      to_visit_time.pop_front();

      visited.insert(current);

      CFace *face = mg.faces[current];
      for (int i=0; i<face->edges.size(); i++) {
        // find punctured edges
        CEdge *edge = face->edges[i];
        EdgeIdType e = edge->id;
        if (_punctured_edges.find(e) != _punctured_edges.end()) {
          const PuncturedEdge& pe = _punctured_edges[e];
          if (current_time >= pe.t) continue; // time ascending order
          
          int echirality = face->edges_chirality[i] * pe.chirality;
          if (current_chirality == echirality) {
            /// find neighbor faces who chontain this edge
            // fprintf(stderr, "fid=%u, found edge eid=%u, t=%f\n", current, e, pe.t);
            for (int j=0; j<edge->contained_faces.size(); j++) {
              if (visited.find(edge->contained_faces[j]->id) == visited.end()) { // not found in visited faces
                to_visit.push_back(edge->contained_faces[j]->id);
                to_visit_chirality.push_back(edge->contained_faces_chirality[j] * current_chirality);
                to_visit_time.push_back(pe.t);
              }
            }
          }
        }
      }

      if (_punctured_faces1.find(current) != _punctured_faces1.end() && 
          _punctured_faces1[current].chirality == current_chirality) 
      {
        std::list<FaceIdType>::iterator it0 = to_visit.begin(); 
        std::list<int>::iterator it1 = to_visit_chirality.begin();
        std::list<double>::iterator it2 = to_visit_time.begin();

        fprintf(stderr,"  {%u, %d, %f}->", it->first, it->second.chirality, 0.0);
        for (int i=0; i<to_visit.size(); i++) {
          fprintf(stderr, "{%u, %d, %f}->", *it0, *it1, *it2);
          it0++; it1++; it2++;
        }
        fprintf(stderr, "{%u, %d, %f}\n", current, current_chirality, 1.0);
        
        related.push_back(current);
      }
    }

#if 0
    fprintf(stderr, "fid=%u, related={", it->first);
    for (int i=0; i<related.size(); i++)
      if (i<related.size()-1)
        fprintf(stderr, "%u, ", related[i]);
      else 
        fprintf(stderr, "%u", related[i]);
    fprintf(stderr, "}\n");
#endif
  }
}

#if 0
void VortexExtractor::Trace()
{
  fprintf(stderr, "tracing, #punctured_elems=%ld.\n", _punctured_elems.size());
  _vortex_objects.clear();

  PuncturedElemMap punctured_elems1 = _punctured_elems; // for final memory cleanup

  while (!_punctured_elems.empty()) {
    /// 1. sort punctured elems into connected ordinary/special ones
    std::list<PuncturedElemMap::iterator> to_erase, to_visit;
    to_visit.push_back(_punctured_elems.begin()); 

    PuncturedElemMap ordinary_pelems, special_pelems; 
    while (!to_visit.empty()) { // depth-first search
      PuncturedElemMap::iterator it = to_visit.front();
      to_visit.pop_front();
      
      if (it->second->visited) continue;
      if (it->second->IsSpecial()) 
        special_pelems[it->first] = it->second;
      else 
        ordinary_pelems[it->first] = it->second; 
      it->second->visited = true; 
      to_erase.push_back(it);

      std::vector<ElemIdType> neighbors = _dataset->GetNeighborIds(it->first); 
      for (int i=0; i<neighbors.size(); i++) {
        ElemIdType id = neighbors[i]; 
        if (id != UINT_MAX && it->second->IsPunctured(i)) {
          PuncturedElemMap::iterator it1 = _punctured_elems.find(id); 
          if (it1 == _punctured_elems.end()) continue; 
          // assert(it1 != _punctured_elems.end()); 
          if (!it1->second->visited)
            to_visit.push_back(it1); 
        }
      }
    }
   
    for (std::list<PuncturedElemMap::iterator>::iterator it = to_erase.begin(); it != to_erase.end(); it ++)
      _punctured_elems.erase(*it);
    to_erase.clear(); 

    // fprintf(stderr, "#ordinary=%ld, #special=%ld\n", ordinary_pelems.size(), special_pelems.size());

    /// 2. trace vortex lines
    VortexObject vortex_object; 
    /// 2.1 clear visited tags
    for (PuncturedElemMap::iterator it = ordinary_pelems.begin(); it != ordinary_pelems.end(); it ++) 
      it->second->visited = false;
    /// 2.2 trace backward and forward
    while (!ordinary_pelems.empty()) {
      PuncturedElemMap::iterator seed = ordinary_pelems.begin();
      to_erase.push_back(seed);

      std::list<double> line; 

      // trace forward (chirality == 1)
      ElemIdType id = seed->first;
      double pos[3];
      bool traced; 
      while (1) {
        traced = false;
        PuncturedElemMap::iterator it = ordinary_pelems.find(id);
        if (it == ordinary_pelems.end() || it->second->visited) break;
        it->second->visited = true;

        std::vector<ElemIdType> neighbors = _dataset->GetNeighborIds(it->first); 
        for (int face=0; face<neighbors.size(); face++) 
          if (it->second->Chirality(face) == 1) {
            if (it != seed) to_erase.push_back(it); 
            it->second->GetPuncturedPoint(face, pos);
            line.push_back(pos[0]); line.push_back(pos[1]); line.push_back(pos[2]);
            if (neighbors[face] != UINT_MAX && special_pelems.find(neighbors[face]) == special_pelems.end()) { // not boundary && not special
              id = neighbors[face]; 
              traced = true;
            }
          }

        if (_dataset->OnBoundary(id)) break;
        if (!traced) break;
      }

      // trace backward (chirality == -1)
      seed->second->visited = false;
      id = seed->first;
      while (1) {
        traced = false;
        PuncturedElemMap::iterator it = ordinary_pelems.find(id);
        if (it == ordinary_pelems.end() || it->second->visited) break;
        it->second->visited = true;

        std::vector<ElemIdType> neighbors = _dataset->GetNeighborIds(it->first); 
        for (int face=0; face<neighbors.size(); face++) 
          if (it->second->Chirality(face) == -1) {
            if (it != seed) to_erase.push_back(it); 
            it->second->GetPuncturedPoint(face, pos);
            line.push_front(pos[2]); line.push_front(pos[1]); line.push_front(pos[0]);
            if (neighbors[face] != UINT_MAX && special_pelems.find(neighbors[face]) == special_pelems.end()) { // not boundary && not special
              id = neighbors[face]; 
              traced = true;
            }
          }

        if (_dataset->OnBoundary(id)) break;
        if (!traced) break;
      }

      for (std::list<PuncturedElemMap::iterator>::iterator it = to_erase.begin(); it != to_erase.end(); it ++)
        ordinary_pelems.erase(*it);
      to_erase.clear();

      vortex_object.AddVortexLine(line);
    }
    _vortex_objects.push_back(vortex_object);
  }
 
  // release memory
  // for (PuncturedElemMap::iterator it = punctured_elems1.begin(); it != punctured_elems1.end(); it ++)
  //   delete it->second;
  
  fprintf(stderr, "#vortex_objects=%ld\n", _vortex_objects.size());
}

bool VortexExtractor::ExtractElem(ElemIdType id)
{
  const int nfaces = _dataset->NrFacesPerElem(); 
  PuncturedElem *pelem = NewPuncturedElem(id);
 
  for (int face=0; face<nfaces; face++) {
    const int nnodes = _dataset->NrNodesPerFace();
    double X[nnodes][3], A[nnodes][3], re[nnodes], im[nnodes]; 

    _dataset->GetFace(id, face, X, A, re, im);

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
    if (fabs(critera)<0.5) continue; // not punctured

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
    if (FindZero(X, re, im, pos))
      pelem->AddPuncturedFace(face, chirality, pos);
    else fprintf(stderr, "WARNING: punctured but singularity not found.\n");
  }

  if (pelem->Punctured()) {
    _punctured_elems[id] = pelem;
    if (pelem->Degree() != 2 && !_dataset->OnBoundary(id))
      fprintf(stderr, "degree=%d\n", pelem->Degree());
    return true;
  } else {
    delete pelem;
    return false;
  }
}
#endif
