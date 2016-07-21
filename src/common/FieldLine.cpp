#include "FieldLine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#if WITH_PROTOBUF
#include "FieldLine.pb.h"
#endif

FieldLine::FieldLine()
{
}

FieldLine::~FieldLine()
{
}

void FieldLine::SerializeToString(std::string& buf) const
{
#if WITH_PROTOBUF
  PBFieldLine fobj; 
  for (int i=0; i<size(); i++) 
    fobj.add_vertices(at(i));
  fobj.SerializeToString(&buf);
#endif
}

bool FieldLine::UnserializeFromString(const std::string& buf) 
{
#if WITH_PROTOBUF
  PBFieldLine fobj; 
  if (!fobj.ParseFromString(buf)) 
    return false; 

  clear(); 

  for (int i=0; i<fobj.vertices_size(); i++) 
    push_back(fobj.vertices(i)); 

  return true;
#else
  return false;
#endif
}

////////////
// File layout: 
//   num_lines (1*size_t) 
//   sizes (num_lines*size_t)
//   buffers
////////////
void WriteFieldLines(const std::string& filename, const std::vector<FieldLine>& objs)
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

bool ReadFieldLines(const std::string& filename, std::vector<FieldLine>& objs)
{
  size_t count;
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) return false;

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

  return true;
}
