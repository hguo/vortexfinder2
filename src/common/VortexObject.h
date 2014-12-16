#ifndef _VORTEX_OBJECT_H
#define _VORTEX_OBJECT_H

#include <string>
#include <list>
#include <vector>
#include <sstream>
#include "def.h"

/* 
 * \class   VortexObject
 * \author  Hanqi Guo
 * \brief   Vortex objects
*/
class VortexObject : public std::vector<std::vector<double> >
{
public:
  VortexObject();
  ~VortexObject(); 

  void AddVortexLine(const std::list<double>& line); 

  // (un)serialization for communication and I/O
  void SerializeToString(std::string& str) const; 
  bool UnserializeFromString(const std::string& str);
}; 

void WriteVortexObjects(const std::string& filename, const std::vector<VortexObject>& objs);

bool ReadVortexOjbects(const std::string& filename, std::vector<VortexObject>& objs); 

#endif
