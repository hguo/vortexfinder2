#ifndef _FIELDLINE_H
#define _FIELDLINE_H

#include <vector>

class FieldLine : public std::vector<double> {
public:
  FieldLine(); 
  ~FieldLine(); 

  // (un)serialization for communication and I/O
  void SerializeToString(std::string& str) const; 
  bool UnserializeFromString(const std::string& str);
};

void WriteFieldLines(const std::string& filename, const std::vector<FieldLine>& objs);

void ReadFieldLines(const std::string& filename, std::vector<FieldLine>& objs); 

#endif
