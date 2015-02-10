#ifndef _PUNCTURE_H
#define _PUNCTURE_H

#include <map>
#include <string>
#include "def.h"

struct PuncturedFace
{
  int chirality;
  double pos[3];
};

struct PuncturedEdge
{
  int chirality; 
  double t; // punctured time
};

struct PuncturedCell
{
  int chiralities[6]; // chiralities on faces
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
