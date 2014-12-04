#include "vortex_object.h"
#include "vortex_object.pb.h"

VortexObject::VortexObject()
{
}

VortexObject::~VortexObject()
{
}

void VortexObject::AddVortexLine(const std::list<double>& line)
{
  std::vector<double> line1;
  line1.reserve(line.size()); 

  for (std::list<double>::const_iterator it = line.begin(); it != line.end(); it ++) 
    line1.push_back(*it);

  push_back(line1);
}

void VortexObject::SerializeToString(std::string& buf) const
{
  PBVortexObject pbobj; 
  for (int i=0; i<size(); i++) {
    PBVortexCoreLine *pbline = pbobj.add_lines(); 
    for (int j=0; j<at(i).size(); j++) 
      pbline->add_vertices(at(i)[j]); 
  }
  pbobj.SerializeToString(&buf);
}

bool VortexObject::UnserializeFromString(const std::string& buf)
{
  PBVortexObject pbobj; 
  if (!pbobj.ParseFromString(buf)) 
    return false; 

  clear(); 
  
  for (int i=0; i<pbobj.lines_size(); i++) {
    std::vector<double> line;
    PBVortexCoreLine *pbline = pbobj.mutable_lines(i); 
    for (int j=0; j<pbline->vertices_size(); j++) 
      line.push_back(pbline->vertices(j)); 
    push_back(line);
  }

  return true; 
}
