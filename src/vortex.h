#ifndef _VORTEX_H
#define _VORTEX_H

#include <string>
#include <list>
#include <vector>

typedef unsigned int ElemIdType;

////////////////////////////////
// an element with punctured faces and connectivities
template <typename T=double, int NrFaces=4, int NrDims=3>
struct VortexItem {
  ElemIdType elem_id;
  std::bitset<NrFaces*2> bits;
  std::vector<T> pos;
  mutable bool visited; 

  VortexItem() : pos(NrFaces*NrDims), visited(false) {}

  bool operator<(const VortexItem<T, NrFaces, NrDims> &rhs) const {return elem_id < rhs.elem_id;}

  bool Valid() const {return bits.any();} // returns false if no punctured faces

  int Chirality(int face) const {
    if (!bits[face]) return 0; // face not punctured
    else return bits[face+4] ? 1 : -1; 
  }
  
  void SetChirality(int face, int chirality) {if (chirality==1) bits[face+NrFaces] = 1;}
  bool IsPunctured(int face) const {return bits[face];}
  void SetPuncturedFace(int face) {bits[face] = 1;}
  void RemovePuncturedFace(int face) {bits[face] = 0; bits[face+NrFaces] = 0;} 
  bool IsSpecial() const {return Degree()>2;}
  
  void SetPuncturedPoint(int face, const T* p) {
    for (int i=0; i<NrDims; i++) 
      pos[face*NrDims+i] = p[i];
  }
  
  void GetPuncturedPoint(int face, T* p) const {
    for (int i=0; i<NrDims; i++) 
      p[i] = pos[face*NrDims+i];
  }

  int Degree() const {
    int deg = 0; 
    for (int i=0; i<NrFaces; i++) 
      deg += bits[i]; 
    return deg; 
  }
};


////////////////////////////////
template <typename T=double, int NrFaces=4, int NrDims=3>
class VortexMap : public std::map<ElemIdType, VortexItem<T, NrFaces, NrDims> > 
{
};

////////////////////////////////
template <typename T=double, int NrFaces=4, int NrDims=3> 
class VortexObject : public std::list<std::list<T> >
{
public:
  VortexObject() {}
  ~VortexObject() {}

  void Serialize(std::string& str);
  bool Unserialize(const std::string& str);

private:
  // VortexMap<T, NrFaces, NrDims> _map;
}; 

#endif
