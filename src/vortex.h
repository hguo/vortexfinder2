#ifndef _VORTEX_H
#define _VORTEX_H

#include <string>
#include <list>

class VortexObject 
{
public:
  VortexObject(); 
  ~VortexObject();

  void SerializeToString(std::string& str);
  bool ParseFromString(const std::string& str);

protected:
  typedef std::list<float> PointList;
  std::list<PointList> _vortices; 
}; 

#endif
