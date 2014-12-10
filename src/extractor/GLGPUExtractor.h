#ifndef _GLGPUEXTRACTOR_H
#define _GLGPUEXTRACTOR_H

#include <map>
#include <list>
#include "Extractor.h"
  
typedef struct {
  float x, y, z; 
} point_t;

class GLGPUVortexExtractor : public VortexExtractor {
  typedef struct {
    float x, y, z; 
    int flag; 
  } punctured_point_t;

public:
  GLGPUVortexExtractor(); 
  ~GLGPUVortexExtractor();

  void SetDataset(const GLDataset *ds); 

  void Extract();
  void Trace(); 
private:
  void solve(int x, int y, int z, int face);
  void trace(std::map<int, punctured_point_t>::iterator it, std::list<std::map<int, punctured_point_t>::iterator>& traversed, bool dir, bool seed); 
  
  const std::list<std::list<point_t> >& cores() const {return _cores;} 

  void id2cell(int id, int *x, int *y, int *z);
  int cell2id(int x, int y, int z);

  void id2face(int id, int *f, int *x, int *y, int *z);
  int face2id(int f, int x, int y, int z); 

  float gauge(int *x0, int *x1) const;  

private: 
  const GLGPUDataset *_ds; 

  std::map<int, punctured_point_t> _points;  // <faceId, point>
  std::list<std::list<point_t> > _cores;   
}; 

#endif
