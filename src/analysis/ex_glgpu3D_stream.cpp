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
#include "io/GLGPU3DDataset.h"
#include "extractor/Extractor.h"

#include <thread>
#include <chrono>

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
typedef struct {
  int tag;
  vfgpu_hdr_t hdr;
  std::vector<vfgpu_pf_t> pfs;
} task_t;

std::queue<task_t> Q;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

void* exec_thread(void*)
{
  GLGPU3DDataset *ds = NULL;
  VortexExtractor *ex = NULL;
  task_t task;
  
  while (1) {
    pthread_mutex_lock(&mutex);
    while (Q.empty()) {
      pthread_cond_wait(&cond, &mutex);
    }
    task = Q.front();
    Q.pop();
    pthread_mutex_unlock(&mutex);
    if (task.tag != 0) break; // all done

    if (ds == NULL) {
      GLHeader h;
      h.ndims = 3;
      memcpy(h.dims, task.hdr.d, sizeof(int)*3);
      memcpy(h.pbc, task.hdr.pbc, sizeof(int)*3);
      memcpy(h.lengths, task.hdr.lengths, sizeof(float)*3);
      memcpy(h.origins, task.hdr.origins, sizeof(float)*3);
      memcpy(h.cell_lengths, task.hdr.cell_lengths, sizeof(float)*3);

      ds = new GLGPU3DDataset;
      ds->SetHeader(h);
      ds->SetMeshType(GLGPU3D_MESH_HEX); // TODO
      ds->BuildMeshGraph();

      ex = new VortexExtractor;
      ex->SetDataset(ds);
    }
    
    ex->Clear();
    for (int i=0; i<task.pfs.size(); i++) {
      vfgpu_pf_t &pf = task.pfs[i];
      int chirality = pf.fid_and_chirality & 0x80000000 ? 1 : -1;
      int fid = pf.fid_and_chirality & 0x7fffffff;
      ex->AddPuncturedFace(fid, 0, chirality, pf.pos);
    }
    ex->TraceOverSpace(0);

    std::vector<VortexLine> vlines = ex->GetVortexLines();
    fprintf(stderr, "frame=%d, #pfs=%d, #vlines=%d\n", 
        task.hdr.frame, (int)task.pfs.size(), (int)vlines.size());

    std::stringstream ss;
    ss << "vlines-" << task.hdr.frame << ".vtk";
    SaveVortexLinesVTK(vlines, ss.str());
  }

  delete ds;
  delete ex;
  return NULL;
}

/////////////////
int main(int argc, char **argv)
{
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

  const int nthreads = std::thread::hardware_concurrency() - 1;

  pthread_t threads[nthreads];
  for (int i=0; i<nthreads; i++)
    pthread_create(&threads[i], NULL, exec_thread, NULL);

  while (!feof(fp)) {
    fread(&hdr, sizeof(vfgpu_hdr_t), 1, fp);
    fread(&pfcount, sizeof(int), 1, fp);
    if (pfcount > 0) {
      task_t task;
      task.tag = 0;
      task.hdr = hdr;
      task.pfs.resize(pfcount);
      fread(task.pfs.data(), sizeof(vfgpu_pf_t), pfcount, fp);

      pthread_mutex_lock(&mutex);
      Q.push(task);
      pthread_cond_signal(&cond);
      pthread_mutex_unlock(&mutex);
    }
  }

  // exit threads
  task_t task_all_done;
  memset(&task_all_done, 0, sizeof(task_t));
  task_all_done.tag = 1;
  for (int i=0; i<nthreads; i++) {
    pthread_mutex_lock(&mutex);
    Q.push(task_all_done);
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
  }
  for (int i=0; i<nthreads; i++) 
    pthread_join(threads[i], NULL);

  fclose(fp);

  fprintf(stderr, "exiting...\n");
  return 0;
}
