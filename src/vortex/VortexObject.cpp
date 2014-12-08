#include "VortexObject.h"
#include "VortexObject.pb.h"

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

////////////
// File layout: 
//   num_objs (1*size_t) 
//   sizes (num_objs*size_t)
//   buffers
////////////
void WriteVortexObjects(const std::string& filename, const std::vector<VortexObject>& objs)
{
  FILE *fp = fopen(filename.c_str(), "wb");
  assert(fp); 
  
  size_t count = objs.size(); 
  size_t *sizes = (size_t*)malloc(count*sizeof(size_t)); 

  fwrite(&count, sizeof(size_t), 1, fp); 
  fwrite(sizes, sizeof(size_t), count, fp); // dummy write sizes

  for (int i=0; i<objs.size(); i++) {
    std::string buf; 
    objs[i].SerializeToString(buf);
    sizes[i] = buf.size(); 
    fwrite(buf.data(), 1, buf.size(), fp); 
  }
 
  fseek(fp, sizeof(size_t), SEEK_SET);
  fwrite(sizes, sizeof(size_t), count, fp); // actual write sizes

  free(sizes); 
  fclose(fp);
}

void ReadVortexOjbects(const std::string& filename, std::vector<VortexObject>& objs)
{
  size_t count;
  FILE *fp = fopen(filename.c_str(), "rb");
  assert(fp);

  fread(&count, sizeof(size_t), 1, fp);
  size_t *sizes = (size_t*)malloc(count*sizeof(size_t));
  fread(sizes, sizeof(size_t), count, fp); 

  objs.resize(count);
  for (int i=0; i<objs.size(); i++) {
    std::string buf; 
    buf.resize(sizes[i]); 

    fread((char*)buf.data(), 1, buf.size(), fp); 
    
    objs[i].UnserializeFromString(buf);
  }

  free(sizes); 
  fclose(fp); 
}
