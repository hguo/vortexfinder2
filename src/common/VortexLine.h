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
struct VortexLine : public std::vector<double>
{
  VortexLine();
  ~VortexLine(); 

  void ToBezier();
  void ToRegular(const double d=0.1);

  void Flattern(const double O[3], const double L[3]);
  void Unflattern(const double O[3], const double L[3]);

  int id, gid;
  int timestep;
  bool is_bezier;

  unsigned char r, g, b;
};

bool SerializeVortexLines(const std::vector<VortexLine>& lines, const std::string& info, std::string& buf);
bool UnserializeVortexLines(std::vector<VortexLine>& lines, std::string& info, const std::string& buf);

bool SaveVortexLines(const std::vector<VortexLine>& lines, const std::string& info, const std::string& filename);
bool LoadVortexLines(std::vector<VortexLine>& lines, std::string& info, const std::string& filename); 

#endif
