#ifndef _VORTEX_LINE_H
#define _VORTEX_LINE_H

#include <string>
#include <list>
#include <vector>
#include <sstream>
#include "def.h"

/* 
 * \class   VortexLine
 * \author  Hanqi Guo
 * \brief   Vortex objects
*/
struct VortexLine : public std::vector<float>
{
  VortexLine();
  ~VortexLine(); 

  void ToBezier();
  void ToRegular(const float d=0.1);

  void Flattern(const float O[3], const float L[3]);
  void Unflattern(const float O[3], const float L[3]);

  void BoundingBox(float LB[3], float UB[3]) const;
  float MaxExtent() const;

  friend float MinimumDist(const VortexLine& l0, const VortexLine& l1);
  friend float CrossingPoint(const VortexLine& l0, const VortexLine& l1, float X[3]);
  friend float Area(const VortexLine& l0, const VortexLine& l1);

  int id, gid;
  int timestep;
  float time;
  bool is_bezier;
  bool is_loop;

  unsigned char r, g, b;
};

bool SerializeVortexLines(const std::vector<VortexLine>& lines, const std::string& info, std::string& buf);
bool UnserializeVortexLines(std::vector<VortexLine>& lines, std::string& info, const std::string& buf);

bool SaveVortexLines(const std::vector<VortexLine>& lines, const std::string& info, const std::string& filename);
bool LoadVortexLines(std::vector<VortexLine>& lines, std::string& info, const std::string& filename);

bool SaveVortexLinesVTK(const std::vector<VortexLine>& lines, const std::string& filename);

#endif
