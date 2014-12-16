#ifndef _FIELDLINE_H
#define _FIELDLINE_H

#include <vector>
#include <list>

class FieldLine : public std::vector<double> {
public:
  FieldLine();
  FieldLine(const std::list<double>&);
  ~FieldLine(); 

  // (un)serialization for communication and I/O
  void SerializeToString(std::string& str) const; 
  bool UnserializeFromString(const std::string& str);
};

void WriteFieldLines(const std::string& filename, const std::vector<FieldLine>& objs);

bool ReadFieldLines(const std::string& filename, std::vector<FieldLine>& objs); 

#endif
