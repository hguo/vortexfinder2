#ifndef _PUNCTURED_ELEM_H
#define _PUNCTURED_ELEM_H

#include <vector>
#include <bitset>
#include <map>
#include "def.h"

class PuncturedElem; 
typedef std::map<ElemIdType, PuncturedElem*> PuncturedElemMap;

class PuncturedElem {
public:
  mutable bool visited; // for traversal

public:
  PuncturedElem() : visited(false) {}
  virtual ~PuncturedElem() {}

  void Init() {
    _pos.resize(NrFaces()*NrDims());
  }

  void SetElemId(ElemIdType id) {_elem_id = id;}
  ElemIdType ElemId() const {return _elem_id;}

  bool Punctured() const {return _bits.any();} // returns false if no punctured faces

  int Chirality(int face) const {
    if (!_bits[face]) return 0; // face not punctured
    else return _bits[face+NrFaces()] ? 1 : -1; 
  }
  
  void AddPuncturedFace(int face, int chirality, const double *p) {
    _bits[face] = 1; 
    if (chirality>0) 
      _bits[face+NrFaces()] = 1;
    for (int i=0; i<NrDims(); i++) 
      _pos[face*NrDims()+i] = p[i];
  }
  
  void GetPuncturedPoint(int face, double* p) const {
    for (int i=0; i<NrDims(); i++) 
      p[i] = _pos[face*NrDims()+i];
  }
  
  void RemovePuncturedFace(int face) {_bits[face] = 0; _bits[face+NrFaces()] = 0;} 
  bool IsPunctured(int face) const {return _bits[face];}
  bool IsSpecial() const {return Degree()>2;}
  
  int Degree() const {
    int deg = 0; 
    for (int i=0; i<NrFaces(); i++) 
      deg += _bits[i]; 
    return deg; 
  }

  void PrintInfo() const {
    fprintf(stderr, "%s, deg=%d\n", _bits.to_string().c_str(), Degree());
  }

private:
  ElemIdType _elem_id;
  std::bitset<16> _bits;
  std::vector<double> _pos; // punctured position
  std::vector<PuncturedElem*> _neighbors;

protected:
  virtual int NrFaces() const = 0;
  virtual int NrDims() const = 0;
};

// Triangle
class PuncturedElemTri : public PuncturedElem
{
  int NrDims() const {return 2;}
  int NrFaces() const {return 3;}
};

// Quadrilateral
class PuncturedElemQuad : public PuncturedElem
{
  int NrDims() const {return 2;}
  int NrFaces() const {return 4;}
};

// Tetrahedron
class PuncturedElemTet : public PuncturedElem
{
  int NrDims() const {return 3;}
  int NrFaces() const {return 4;}
};

// Hexahedron
class PuncturedElemHex : public PuncturedElem
{
  int NrDims() const {return 3;}
  int NrFaces() const {return 6;}
};



#if 0

/* 
 * \class   PuncturedElem
 * \author  Hanqi Guo
 * \brief   An element with punctured faces
*/
template <typename T=double, int NrFaces=4, int NrDims=3>
struct PuncturedElem {
  ElemIdType elem_id;
  std::bitset<NrFaces*2> bits;
  std::vector<T> pos;
  mutable bool visited; 

  PuncturedElem() : pos(NrFaces*NrDims), visited(false) {}

  bool operator<(const PuncturedElem<T, NrFaces, NrDims> &rhs) const {return elem_id < rhs.elem_id;}

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


/* 
 * \class   PuncturedElemMap
 * \author  Hanqi Guo
 * \brief   A dictionary of punctured elems
*/
template <typename T=double, int NrFaces=4, int NrDims=3>
class PuncturedElemMap : public std::map<ElemIdType, PuncturedElem<T, NrFaces, NrDims> > 
{
};

#endif

#endif
