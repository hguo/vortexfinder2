#include "Extractor.h"
#include <cassert>

VortexExtractor::VortexExtractor() :
  _gauge(false)
{

}

VortexExtractor::~VortexExtractor()
{

}

void VortexExtractor::SetGaugeTransformation(bool g)
{
  _gauge = g; 
}

void VortexExtractor::WriteVortexObjects(const std::string& filename)
{
  ::WriteVortexObjects(filename, _vortex_objects); 
}

void VortexExtractor::Trace()
{
  fprintf(stderr, "tracing.\n");
  _vortex_objects.clear();

  while (!_punctured_elems.empty()) {
    /// 1. sort punctured elems into connected ordinary/special ones
    std::list<PuncturedElemMap::iterator> to_erase, to_visit;
    to_visit.push_back(_punctured_elems.begin()); 

    PuncturedElemMap ordinary_pelems, special_pelems; 
    while (!to_visit.empty()) { // depth-first search
      PuncturedElemMap::iterator it = to_visit.front();
      to_visit.pop_front();
      
      if (it->second->IsSpecial()) 
        special_pelems[it->first] = it->second;
      else 
        ordinary_pelems[it->first] = it->second; 
      it->second->visited = true; 
      to_erase.push_back(it);

      std::vector<unsigned int> neighbors = Neighbors(it->first); 
      for (int i=0; i<neighbors.size(); i++) {
        unsigned int id = neighbors[i]; 
        if (id != UINT_MAX && it->second->IsPunctured(i)) {
          PuncturedElemMap::iterator it1 = _punctured_elems.find(id); 
          assert(it1 != _punctured_elems.end()); 
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
      unsigned int id = seed->first;
      double pos[3];
      bool traced; 
      while (1) {
        traced = false;
        PuncturedElemMap::iterator it = ordinary_pelems.find(id);
        if (it == ordinary_pelems.end() || it->second->visited) break;

        std::vector<unsigned int> neighbors = Neighbors(it->first); 
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

        if (!traced) break;
      }

      // trace backward (chirality == -1)
      line.pop_front(); line.pop_front(); line.pop_front(); // remove the seed point
      id = seed->first;
      while (1) {
        traced = false;
        PuncturedElemMap::iterator it = ordinary_pelems.find(id);
        if (it == ordinary_pelems.end() || it->second->visited) break;

        std::vector<unsigned int> neighbors = Neighbors(it->first); 
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

        if (!traced) break;
      }

      for (std::list<PuncturedElemMap::iterator>::iterator it = to_erase.begin(); it != to_erase.end(); it ++)
        ordinary_pelems.erase(*it);
      to_erase.clear();

      vortex_object.AddVortexLine(line);
    }
    _vortex_objects.push_back(vortex_object);
  }
}
