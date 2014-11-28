#ifndef _VORTEX_H
#define _VORTEX_H

#include <string>
#include <list>
#include <vector>

class VortexObject : public std::vector<std::list<float> >
{
public:
  VortexObject(); 
  ~VortexObject();

  void SerializeToString(std::string& str);
  bool ParseFromString(const std::string& str);

  int AddLine();  
  int AddLine(const std::list<float> &line);
  std::list<float>& GetLine(int i) {return (*this)[i];}
  
  int NrLines() const {return size();}
}; 

#endif
