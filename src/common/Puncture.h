#ifndef _PUNCTURE_H
#define _PUNCTURE_H

#include <map>
#include <bitset>
#include "def.h"

struct PuncturedFace
{
  ChiralityType chirality;
  float pos[3];
  float cond; // condition number
};

struct PuncturedEdge
{
  ChiralityType chirality; 
  float t; // punctured time
};

struct PuncturedCell
{
  ChiralityType Chirality(int face) const {
    if (!p[face]) return 0; 
    else return c[face] ? 1 : -1;
  }

  void SetChirality(int face, ChiralityType chirality) {
    p[face] = 1;
    if (chirality>0) c[face] = 1;
  }

  bool IsSpecial() const {return Degree()>2;}
  // bool IsSpecial() const {return Degree() > 0 && Degree() != 2;}
  int Degree() const {return p.count();}

private:
  std::bitset<8> p, c;
  // std::bitset<16> bits;
  // int chiralities[6]; // chiralities on faces
};

//////// I/O for faces
bool SerializePuncturedFaces(const std::map<FaceIdType, PuncturedFace> m, std::string &buf);
bool UnserializePuncturedFaces(std::map<FaceIdType, PuncturedFace> &m, const std::string &buf);

bool SavePuncturedFaces(const std::map<FaceIdType, PuncturedFace> m, const std::string &filename);
bool LoadPuncturedFaces(std::map<FaceIdType, PuncturedFace> &m, const std::string &filename);

//////// I/O for edges
bool SerializePuncturedEdges(const std::map<EdgeIdType, PuncturedEdge> m, std::string &buf);
bool UnserializePuncturedEdges(std::map<EdgeIdType, PuncturedEdge> &m, const std::string &buf);

bool SavePuncturedEdges(const std::map<EdgeIdType, PuncturedEdge> m, const std::string &filename);
bool LoadPuncturedEdges(std::map<EdgeIdType, PuncturedEdge> &m, const std::string &filename);

#endif
