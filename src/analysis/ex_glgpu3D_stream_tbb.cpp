#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <vector>
#include <queue>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <tbb/flow_graph.h>
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
  unsigned int fid_and_chirality; 
  float pos[3];
} vfgpu_pf_t; // punctured faces from GPU output, 16 bytes

typedef struct {
  unsigned int eid;
  signed char chirality;
} vfgpu_pe_t;

typedef struct {
  int frame;
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
struct extract {
  const vfgpu_hdr_t hdr;
  const std::vector<vfgpu_pf_t> pfs;

  extract(const vfgpu_hdr_t& h, const std::vector<vfgpu_pf_t>& p) : 
    hdr(h), pfs(p)
  {
  }

  void operator()(tbb::flow::continue_msg) const {
    GLGPU3DDataset *ds = NULL;
    VortexExtractor *ex = NULL;
    
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
    
    ex->Clear();
    for (int i=0; i<pfs.size(); i++) {
      const vfgpu_pf_t &pf = pfs[i];
      int chirality = pf.fid_and_chirality & 0x80000000 ? 1 : -1;
      int fid = pf.fid_and_chirality & 0x7fffffff;
      ex->AddPuncturedFace(fid, 0, chirality, pf.pos);
    }
    ex->TraceOverSpace(0);

    std::vector<VortexLine> vlines = ex->GetVortexLines();
    fprintf(stderr, "frame=%d, #pfs=%d, #vlines=%d\n", 
        hdr.frame, (int)pfs.size(), (int)vlines.size());

    std::stringstream ss;
    ss << "vlines-" << hdr.frame << ".vtk";
    SaveVortexLinesVTK(vlines, ss.str());

    delete ds;
    delete ex;
  }
}; 

/////////////////
int main(int argc, char **argv)
{
  using namespace tbb::flow;
  graph g;

  vfgpu_hdr_t hdr;
  int pfcount, pfcount_max=0;

  std::string filename; 
  if (argc > 1) filename = argv[1];
  else filename = "/tmp/glgpu.fifo";

  FILE *fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    fprintf(stderr, "cannot open pipe %s\n", filename.c_str());
    exit(1);
  }
  assert(fp);

  while (!feof(fp)) {
    fread(&hdr, sizeof(vfgpu_hdr_t), 1, fp);
    fread(&pfcount, sizeof(int), 1, fp);
    

    if (pfcount > 0) {
      std::vector<vfgpu_pf_t> pfs;
      pfs.resize(pfcount);
      fread(pfs.data(), sizeof(vfgpu_pf_t), pfcount, fp);
      
      continue_node<continue_msg> *e = new continue_node<continue_msg>(g, extract(hdr, pfs));
      e->try_put(continue_msg());
    }
  }

  fclose(fp);

  g.wait_for_all();
  fprintf(stderr, "exiting...\n");
  return 0;
}
