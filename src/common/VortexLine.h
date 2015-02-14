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

  int id;
  int timestep;
}; 

bool SerializeVortexLines(const std::vector<VortexLine>& lines, std::string& buf);
bool UnserializeVortexLines(std::vector<VortexLine>& lines, const std::string& buf);

bool SaveVortexLines(const std::vector<VortexLine>& lines, const std::string& filename);
bool LoadVortexLines(std::vector<VortexLine>& lines, const std::string& filename); 

#endif
