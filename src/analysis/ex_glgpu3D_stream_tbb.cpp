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
#include <tbb/mutex.h>
#include <tbb/flow_graph.h>
#include <tbb/concurrent_unordered_map.h>
#include "io/GLGPU3DDataset.h"
#include "extractor/Extractor.h"

#if WITH_ROCKSDB
#include <rocksdb/db.h>
#endif

enum {
  VFGPU_MSG_PF = 0,
  VFGPU_MSG_PE = 1
};

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

typedef unsigned int vfgpu_pe_t; 

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

tbb::concurrent_unordered_map<int, vfgpu_hdr_t> hdrs_all;
tbb::concurrent_unordered_map<int, std::vector<vfgpu_pf_t> > pfs_all;
tbb::concurrent_unordered_map<int, std::vector<VortexObject> > vobjs_all;
tbb::concurrent_unordered_map<int, std::vector<VortexLine> > vlines_all;
tbb::concurrent_unordered_map<std::pair<int, int>, std::vector<vfgpu_pe_t> > pes_all; // released on exit
std::map<int, tbb::flow::continue_node<tbb::flow::continue_msg>* > extract_tasks; 
std::map<std::pair<int, int>, tbb::flow::continue_node<tbb::flow::continue_msg>* > track_tasks;
VortexTransition vt;

tbb::concurrent_unordered_map<int, int> frame_counter;  // used to count how many times a frame is referenced by trackers
// tbb::concurrent_unordered_map<int, tbb::mutex> frame_mutexes;
static const int max_buffered_frames = 256;
static int num_buffered_frames = 0;

static const int meshType = VFGPU_MESH_TET;
static std::string infile;

#ifdef WITH_ROCKSDB
static rocksdb::DB* db;
#endif

static GLHeader conv_hdr(const vfgpu_hdr_t& hdr) {
  GLHeader h;
  h.ndims = 3;
  memcpy(h.dims, hdr.d, sizeof(int)*3);
  memcpy(h.pbc, hdr.pbc, sizeof(int)*3);
  memcpy(h.lengths, hdr.lengths, sizeof(float)*3);
  memcpy(h.origins, hdr.origins, sizeof(float)*3);
  memcpy(h.cell_lengths, hdr.cell_lengths, sizeof(float)*3);
  return h;
}

static void write_vlines(int frame, const std::vector<VortexLine>& vlines)
{
  std::stringstream ss;
  std::string buf;
  diy::serialize(vlines, buf);
  ss << "v." << frame;
  db->Put(rocksdb::WriteOptions(), ss.str(), buf);
}

static void write_mat(int f0, int f1, const VortexTransitionMatrix& mat)
{
  std::stringstream ss;
  ss << "m." << f0 << "." << f1;
  std::string buf;
  diy::serialize(mat, buf);
  db->Put(rocksdb::WriteOptions(), ss.str(), buf);
}

/////////////////
struct extract {
  int frame;
  extract(int frame_) : frame(frame_) {}

  void operator()(tbb::flow::continue_msg) const {
    const vfgpu_hdr_t& hdr = hdrs_all[frame];
    const std::vector<vfgpu_pf_t>& pfs = pfs_all[frame];
    GLHeader h = conv_hdr(hdrs_all[frame]);

    GLGPU3DDataset *ds = new GLGPU3DDataset;
    ds->SetHeader(h);
    ds->SetMeshType(meshType); // TODO
    ds->BuildMeshGraph();

    VortexExtractor *ex = new VortexExtractor;
    ex->SetDataset(ds);
    
    ex->Clear();
    for (int i=0; i<pfs.size(); i++) {
      const vfgpu_pf_t &pf = pfs[i];
      int chirality = pf.fid_and_chirality & 0x80000000 ? 1 : -1;
      int fid = pf.fid_and_chirality & 0x7fffffff;
      ex->AddPuncturedFace(fid, 0, chirality, pf.pos);
    }
    ex->TraceOverSpace(0);

    vobjs_all[hdr.frame] = ex->GetVortexObjects(0);

    std::vector<VortexLine> vlines = ex->GetVortexLines();
    vlines_all[hdr.frame] = vlines;
    // for (int i=0; i<vlines.size(); i++) 
    //   vlines[i].ToBezier();

    // write_vlines(hdr.frame, vlines);

    delete ds;
    delete ex;

    fprintf(stderr, "frame=%d, #pfs=%d, #vlines=%d\n", 
        hdr.frame, (int)pfs.size(), (int)vlines.size());
    
    __sync_fetch_and_sub(&num_buffered_frames, 1);
  }
}; 

/////////////////
struct track {
  const std::pair<int, int> interval;
  const int f0, f1;
  track(const std::pair<int, int> f) : interval(f), f0(f.first), f1(f.second) {}

  void operator()(tbb::flow::continue_msg) const {
    const vfgpu_hdr_t& hdr0 = hdrs_all[f0], 
                       hdr1 = hdrs_all[f1];
    GLHeader h0 = conv_hdr(hdr0), 
             h1 = conv_hdr(hdr1);
    const std::vector<vfgpu_pf_t>& pfs0 = pfs_all[f0], 
                                   pfs1 = pfs_all[f1];
    const std::vector<vfgpu_pe_t>& pes = pes_all[interval];
    const std::vector<VortexObject>& vobjs0 = vobjs_all[f0],
                                     vobjs1 = vobjs_all[f1];
    std::vector<VortexLine>& vlines0 = vlines_all[f0], 
                             vlines1 = vlines_all[f1];

    GLGPU3DDataset *ds = new GLGPU3DDataset;
    ds->SetHeader(h0);
    ds->SetMeshType(meshType); // TODO
    ds->BuildMeshGraph();

    VortexExtractor *ex = new VortexExtractor;
    ex->SetDataset(ds);
    
    for (int i=0; i<pfs0.size(); i++) {
      const vfgpu_pf_t &pf = pfs0[i];
      int chirality = pf.fid_and_chirality & 0x80000000 ? 1 : -1;
      int fid = pf.fid_and_chirality & 0x7fffffff;
      ex->AddPuncturedFace(fid, 0, chirality, pf.pos);
    }
    
    for (int i=0; i<pfs1.size(); i++) {
      const vfgpu_pf_t &pf = pfs1[i];
      int chirality = pf.fid_and_chirality & 0x80000000 ? 1 : -1;
      int fid = pf.fid_and_chirality & 0x7fffffff;
      ex->AddPuncturedFace(fid, 1, chirality, pf.pos);
    }

    for (int i=0; i<pes.size(); i++) {
      const vfgpu_pe_t &pe = pes[i];
      int chirality = pe & 0x80000000 ? 1 : -1;
      int eid = pe & 0x7fffffff;
      ex->AddPuncturedEdge(eid, chirality, 0);
    }

    ex->SetVortexObjects(vobjs0, 0);
    ex->SetVortexObjects(vobjs1, 1);
    VortexTransitionMatrix mat = ex->TraceOverTime();
    mat.SetInterval(interval);
    mat.Modularize();
    vt.AddMatrix(mat);

    // compute moving speed
#if 0
    for (int i=0; i<mat.NModules(); i++) {
      int e;
      std::set<int> lhs, rhs;
      mat.GetModule(i, lhs, rhs, e);
      if (e == VORTEX_EVENT_DUMMY) {
        int l = *lhs.begin(), r = *rhs.begin();
        VortexLine& l0 = vlines0[l], l1 = vlines1[r];
        float area = Area(l0, l1);
        l0.moving_speed = area / (interval.second - interval.first);
      }
    }
#endif

    delete ex;
    delete ds;
   
    write_vlines(f0, vlines0);
    write_mat(f0, f1, mat);
    
    fprintf(stderr, "interval={%d, %d}, #pfs0=%d, #pfs1=%d, #pes=%d\n", 
        interval.first, interval.second, (int)pfs0.size(), (int)pfs1.size(), (int)pes.size());
    
    // release resources
    pes_all[interval].clear();
    int &fc0 = frame_counter[f0], 
        &fc1 = frame_counter[f1];
    __sync_fetch_and_add(&fc0, 1);
    __sync_fetch_and_add(&fc1, 1);
    if (fc0 == 2) {
      pfs_all[fc0] = std::vector<vfgpu_pf_t>();
      vobjs_all[fc0] = std::vector<VortexObject>();
      vlines_all[fc0] = std::vector<VortexLine>();
    }
    if (fc1 == 2) {
      pfs_all[fc1] = std::vector<vfgpu_pf_t>();
      vobjs_all[fc1] = std::vector<VortexObject>();
      vlines_all[fc1] = std::vector<VortexLine>();
    }
    
    __sync_fetch_and_sub(&num_buffered_frames, 1);
  }
};

/////////////////
int main(int argc, char **argv)
{
  if (argc < 2) return 1;
  infile = argv[1];
  
  FILE *fp = fopen(infile.c_str(), "rb");
  if (!fp) return 1;

#if WITH_ROCKSDB
  std::string dbname = infile + ".rocksdb";

  rocksdb::Options options;
  options.create_if_missing = true;
  // options.compression = rocksdb::kLZ4Compression;
  options.compression = rocksdb::kBZip2Compression;
  // options.write_buffer_size = 64*1024*1024; // 64 MB
  rocksdb::Status status = rocksdb::DB::Open(options, dbname.c_str(), &db);
  assert(status.ok());
#endif

  using namespace tbb::flow;
  graph g;

  int type_msg;
  vfgpu_hdr_t hdr;
  int pfcount, pecount;
  const int max_frames = 1000;
  int frame_count = 0;
  std::vector<int> frames;

  while (!feof(fp)) {
    if (frame_count ++ > max_frames) break;

    // simple flow control
    __sync_fetch_and_add(&num_buffered_frames, 1);
    while (num_buffered_frames >= max_buffered_frames) 
      usleep(100000);

    size_t count = fread(&type_msg, sizeof(int), 1, fp);
    if (count != 1) break;

    if (type_msg == VFGPU_MSG_PF) {
      fread(&hdr, sizeof(vfgpu_hdr_t), 1, fp);
      fread(&pfcount, sizeof(int), 1, fp);
      
      hdrs_all[hdr.frame] = hdr;
      std::vector<vfgpu_pf_t> &pfs = pfs_all[hdr.frame];
      pfs.resize(pfcount);
      fread(pfs.data(), sizeof(vfgpu_pf_t), pfcount, fp);
      
      continue_node<continue_msg> *e = new continue_node<continue_msg>(g, extract(hdr.frame));
      e->try_put(continue_msg());
      extract_tasks[hdr.frame] = e;

      frames.push_back(hdr.frame);
    } else if (type_msg == VFGPU_MSG_PE) {
      std::pair<int, int> interval;
      fread(&interval, sizeof(int), 2, fp);
      fread(&pecount, sizeof(int), 1, fp);

      std::vector<vfgpu_pe_t> &pes = pes_all[interval];
      pes.resize(pecount);
      fread(pes.data(), sizeof(vfgpu_pe_t), pecount, fp);
    
      continue_node<continue_msg> *t = new continue_node<continue_msg>(g, track(interval));
      track_tasks[interval] = t;
     
      make_edge(*extract_tasks[interval.first], *t);
      make_edge(*extract_tasks[interval.second], *t);
    }
  }

  fclose(fp);

  g.wait_for_all();
  
#if WITH_ROCKSDB
  std::string buf;
  
  diy::serialize(frames, buf);
  db->Put(rocksdb::WriteOptions(), "f", buf);

  fprintf(stderr, "constructing sequences...\n");
  vt.SetFrames(frames);
  vt.ConstructSequence();
  vt.PrintSequence();
  diy::serialize(vt, buf);
  db->Put(rocksdb::WriteOptions(), "trans", buf);
  
  delete db;
#endif

  fprintf(stderr, "exiting...\n");
  return 0;
}
