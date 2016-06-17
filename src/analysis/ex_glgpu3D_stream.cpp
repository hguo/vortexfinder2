#include <iostream>
#include <cstdio>
#include <vector>
#include "io/GLGPU3DDataset.h"
#include "extractor/Extractor.h"

enum {
  VFGPU_MESH_HEX = 0,
  VFGPU_MESH_TET,
  VFGPU_MESH_2D
};

enum {
  VFGPU_GAUGE_YZ = 0,
  VFGPU_GAUGE_XZ = 1
};

typedef struct {
  unsigned int fid; 
  signed char chirality;
  float pos[3];
} vfgpu_pf_t; // punctured faces from GPU output, 16 bytes
typedef struct {
  unsigned int eid;
  signed char chirality;
} vfgpu_pe_t;

typedef struct {
  int d[3];
  unsigned int count; // d[0]*d[1]*d[2];
  bool pbc[3];
  float origins[3];
  float lengths[3];
  float cell_lengths[3];
  float B[3];
  float Kx;
} vfgpu_hdr_t;

/////////////////
int main(int argc, char **argv)
{
  vfgpu_hdr_t hdr;
  int pfcount, pfcount_max=0;
  vfgpu_pf_t *pflist = NULL;

  while (!feof(stdin)) {
    fread(&hdr, sizeof(vfgpu_hdr_t), 1, stdin);
    fread(&pfcount, sizeof(int), 1, stdin);
    if (pfcount > pfcount_max)
      pflist = (vfgpu_pf_t*)realloc(pflist, sizeof(vfgpu_pf_t)*pfcount);
    if (pfcount > 0)
      fread(pflist, sizeof(vfgpu_pf_t), pfcount, stdin);
    fprintf(stderr, "pfcount=%d\n", pfcount);
  }

  return 0;
}
