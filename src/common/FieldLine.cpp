#include "FieldLine.h"
#include "FieldLine.pb.h"

FieldLine::FieldLine()
{
}

FieldLine::~FieldLine()
{
}

void FieldLine::SerializeToString(std::string& buf) const
{
  PBFieldLine fobj; 
  for (int i=0; i<size(); i++) 
    fobj.add_vertices(at(i));
  fobj.SerializeToString(&buf);
}

bool FieldLine::UnserializeFromString(const std::string& buf) 
{
  PBFieldLine fobj; 
  if (!fobj.ParseFromString(buf)) 
    return false; 

  clear(); 

  for (int i=0; i<fobj.vertices_size(); i++) 
    push_back(fobj.vertices(i)); 

  return true;
}
