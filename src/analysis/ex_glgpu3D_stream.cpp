#include <iostream>
#include <cstdio>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
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

  GLGPU3DDataset *ds = NULL;
  VortexExtractor *ex = NULL;

  FILE *fp = fopen("/tmp/glgpu.fifo", "rb");

  while (!feof(fp)) {
    fread(&hdr, sizeof(vfgpu_hdr_t), 1, fp);
    fread(&pfcount, sizeof(int), 1, fp);
    if (pfcount > pfcount_max)
      pflist = (vfgpu_pf_t*)realloc(pflist, sizeof(vfgpu_pf_t)*pfcount);
    if (pfcount > 0)
      fread(pflist, sizeof(vfgpu_pf_t), pfcount, fp);
    fprintf(stderr, "pfcount=%d\n", pfcount);

    if (ds == NULL) {
      GLHeader h;
      h.ndims = 3;
      memcpy(h.dims, hdr.d, sizeof(int)*3);
      memcpy(h.pbc, hdr.pbc, sizeof(int)*3);
      memcpy(h.lengths, hdr.lengths, sizeof(float)*3);
      memcpy(h.origins, hdr.origins, sizeof(float)*3);
      memcpy(h.cell_lengths, hdr.cell_lengths, sizeof(float)*3);

      ds = new GLGPU3DDataset;
      ds->SetHeader(h);
      ds->SetMeshType(GLGPU3D_MESH_HEX); // TODO
      ds->BuildMeshGraph();

      ex = new VortexExtractor;
      ex->SetDataset(ds);
    }
  
    ex->Clear();
    for (int i=0; i<pfcount; i++) {
      vfgpu_pf_t &pf = pflist[i];
      ex->AddPuncturedFace(pf.fid, 0, pf.chirality, pf.pos);
    }
    ex->TraceOverSpace(0);
  }

  fclose(fp);
  delete ex;
  delete ds;

  return 0;
}
