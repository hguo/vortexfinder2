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


// Prisms (space-time)
class PuncturedPrismTri : public PuncturedElem
{
  int NrDims() const {return 0;} // we do not need to keep the coordinates
  int NrFaces() const {return 5;} // two real faces and two virtual faces  
};

class PuncturedPrismQuad : public PuncturedElem
{
  int NrDims() const {return 0;} // we do not need to keep the coordinates
  int NrFaces() const {return 6;} // two real faces and 4 virtual faces  
};

#endif
