#ifndef _PUNCTURED_ELEM_H
#define _PUNCTURED_ELEM_H

#include <string>
#include <list>
#include <vector>
#include <bitset>
#include <sstream>
#include "def.h"

typedef unsigned int ElemIdType;

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
