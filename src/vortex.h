#ifndef _VORTEX_H
#define _VORTEX_H

#include <string>
#include <list>
#include <vector>
#include <bitset>
#include <sstream>

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
class VortexObject : public std::vector<std::list<T> >
{
public:
  VortexObject() {}
  ~VortexObject() {}

  // FIXME: use production libraries like protobuf
  void Serialize(std::string& str) const {
    std::ostringstream stream; 
    stream << this->size() << "\t\n";  
    for (typename std::vector<std::list<T> >::const_iterator it = this->begin(); it != this->end(); it ++) 
      stream << it->size() << "\t";
    stream << "\n"; 
    for (typename std::vector<std::list<T> >::const_iterator it = this->begin(); it != this->end(); it ++) { 
      for (typename std::list<T>::const_iterator it1 = it->begin(); it1 != it->end(); it1 ++) 
        stream << *it1 << "\t"; 
      stream << "\n";
    }
    str = stream.str(); 
  }

  bool Unserialize(const std::string& str) {
    std::istringstream stream(str); 
    this->clear();

    size_t n; 
    stream >> n;
    this->resize(n); 

    std::vector<size_t> n_vertices(n);  
    for (size_t i=0; i<n; i++) {
      stream >> n_vertices[i];
      // fprintf(stderr, "n=%d, n_vertices=%d\n", n, n_vertices[i]); 
    }

    for (size_t i=0; i<n; i++) 
      for (size_t j=0; j<n_vertices[i]; j++) {
        T number; 
        stream >> number;
        this->at(i).push_back(number); 
      }

    return true; 
  }

private:
  // VortexMap<T, NrFaces, NrDims> _map;
}; 

#endif
