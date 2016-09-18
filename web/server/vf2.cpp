#include <node.h>
#include <rocksdb/db.h>
#include <string>
#include <sstream>
#include "common/VortexLine.h"
#include "common/VortexTransition.h"
#include "common/Inclusions.h"

using v8::FunctionCallbackInfo;
using v8::Function;
using v8::Array;
using v8::Number;
using v8::Isolate;
using v8::Local;
using v8::Object;
using v8::String;
using v8::Value;
using v8::Exception;
  
typedef struct {
  int timestep;
  float B[3];
  float Kx; // Kx
  float Jxext;
  float V; // voltage
} vfgpu_hdr_t;

typedef struct {
  unsigned char meshtype;
  bool tracking;
  float dt;
  int d[3];
  unsigned int count; // d[0]*d[1]*d[2];
  bool pbc[3];
  float origins[3];
  float lengths[3];
  float cell_lengths[3];
  float zaniso;
} vfgpu_cfg_t;

void LoadDataInfoFromDB(
    const std::string& dbname, 
    vfgpu_cfg_t& cfg,
    std::vector<vfgpu_hdr_t>& hdrs, 
    Inclusions& incs)
{
  rocksdb::DB* db;
  rocksdb::Status s;
  std::string buf;
  
  rocksdb::Options options;
  options.create_if_missing = false;
  s = rocksdb::DB::Open(options, dbname, &db);
  assert(s.ok());
  
  s = db->Get(rocksdb::ReadOptions(), "cfg", &buf);
  if (buf.size() > 0) {
    diy::unserialize(buf, cfg);
  }
  
  s = db->Get(rocksdb::ReadOptions(), "hdrs", &buf);
  if (buf.size() > 0) {
    diy::unserialize(buf, hdrs);
  }

  s = db->Get(rocksdb::ReadOptions(), "inclusions", &buf);
  if (buf.size() > 0) {
    diy::unserialize(buf, incs);
  }

  delete db;
}

bool LoadFrameFromDB(
    const std::string& dbname, 
    int frame, 
    std::vector<VortexLine>& vlines)
{
  fprintf(stderr, "dbname=%s, frame=%d\n", 
      dbname.c_str(), frame);

  rocksdb::DB* db;
  rocksdb::Status s;
  
  rocksdb::Options options;
  options.create_if_missing = false;
  s = rocksdb::DB::Open(options, dbname, &db);
  assert(s.ok());

  std::string buf;
  s = db->Get(rocksdb::ReadOptions(), "trans", &buf);
  if (buf.empty()) {delete db; return false;}
  VortexTransition vt;
  diy::unserialize(buf, vt);

  const int timestep = vt.Frame(frame);
  std::stringstream ss;
  ss << "v." << timestep;
  buf.clear();
  s = db->Get(rocksdb::ReadOptions(), ss.str(), &buf);
  if (buf.empty()) {delete db; return false;}
  diy::unserialize(buf, vlines);

  for (size_t i=0; i<vlines.size(); i++) {
    vlines[i].gid = vt.lvid2gvid(frame, vlines[i].id); // sorry, this is confusing
    vt.SequenceColor(vlines[i].gid, vlines[i].r, vlines[i].g, vlines[i].b);
    // fprintf(stderr, "timestep=%d, lid=%d, gid=%d\n", timestep, vlines[i].id, vlines[i].gid);
  }
  // fprintf(stderr, "#vlines=%d\n", (int)vlines.size());

  delete db;
}

void LoadDataInfo(const FunctionCallbackInfo<Value>& args) {
  Isolate *isolate = args.GetIsolate();

  if (args.Length() < 1) {
    isolate->ThrowException(Exception::TypeError(
          String::NewFromUtf8(isolate, "Wrong number of arguments")));
    return;
  }

  if (!args[0]->IsString()) {
    isolate->ThrowException(Exception::TypeError(
          String::NewFromUtf8(isolate, "Wrong arguments")));
    return;
  }

  // input args
  String::Utf8Value dbname1(args[0]->ToString());
  std::string dbname(*dbname1);

  vfgpu_cfg_t cfg;
  std::vector<vfgpu_hdr_t> hdrs;
  Inclusions incs;
  LoadDataInfoFromDB(dbname, cfg, hdrs, incs);

  // outputs
  Local<Object> jout = Object::New(isolate);
  
  // cfg
  Local<Object> jcfg = Object::New(isolate); 
  {
    Local<Number> jdt = Number::New(isolate, cfg.dt);
    jcfg->Set(String::NewFromUtf8(isolate, "dt"), jdt);
    Local<Number> jNx = Number::New(isolate, cfg.d[0]);
    jcfg->Set(String::NewFromUtf8(isolate, "Nx"), jNx);
    Local<Number> jNy = Number::New(isolate, cfg.d[1]);
    jcfg->Set(String::NewFromUtf8(isolate, "Ny"), jNy);
    Local<Number> jNz = Number::New(isolate, cfg.d[2]);
    jcfg->Set(String::NewFromUtf8(isolate, "Nz"), jNz);
    Local<Number> jOx = Number::New(isolate, cfg.origins[0]);
    jcfg->Set(String::NewFromUtf8(isolate, "Ox"), jOx);
    Local<Number> jOy = Number::New(isolate, cfg.origins[1]);
    jcfg->Set(String::NewFromUtf8(isolate, "Oy"), jOy);
    Local<Number> jOz = Number::New(isolate, cfg.origins[2]);
    jcfg->Set(String::NewFromUtf8(isolate, "Oz"), jOz);
    Local<Number> jLx = Number::New(isolate, cfg.lengths[0]);
    jcfg->Set(String::NewFromUtf8(isolate, "Lx"), jLx);
    Local<Number> jLy = Number::New(isolate, cfg.lengths[1]);
    jcfg->Set(String::NewFromUtf8(isolate, "Ly"), jLy);
    Local<Number> jLz = Number::New(isolate, cfg.lengths[2]);
    jcfg->Set(String::NewFromUtf8(isolate, "Lz"), jLz);
  }
  jout->Set(String::NewFromUtf8(isolate, "cfg"), jcfg);

  // hdrs
  Local<Array> jhdrs = Array::New(isolate);
  for (size_t i=0; i<hdrs.size(); i++) {
    const vfgpu_hdr_t& hdr = hdrs[i];
    Local<Object> jhdr = Object::New(isolate);
    Local<Number> jtimestep = Number::New(isolate, hdr.timestep);
    jhdr->Set(String::NewFromUtf8(isolate, "timestep"), jtimestep);
    Local<Number> jBx = Number::New(isolate, hdr.B[0]);
    jhdr->Set(String::NewFromUtf8(isolate, "Bx"), jBx);
    Local<Number> jBy = Number::New(isolate, hdr.B[1]);
    jhdr->Set(String::NewFromUtf8(isolate, "By"), jBy);
    Local<Number> jBz = Number::New(isolate, hdr.B[2]);
    jhdr->Set(String::NewFromUtf8(isolate, "Bz"), jBz);
    Local<Number> jKx = Number::New(isolate, hdr.Kx);
    jhdr->Set(String::NewFromUtf8(isolate, "Kx"), jKx);
    Local<Number> jJxext = Number::New(isolate, hdr.Jxext);
    jhdr->Set(String::NewFromUtf8(isolate, "Jxext"), jJxext);
    Local<Number> jV = Number::New(isolate, hdr.V);
    jhdr->Set(String::NewFromUtf8(isolate, "V"), jV);
    jhdrs->Set(Number::New(isolate, i), jhdr);
  }
  jout->Set(String::NewFromUtf8(isolate, "hdrs"), jhdrs);
  
  // inclusions
  Local<Array> jincs = Array::New(isolate); 
  for (int i=0; i<incs.Count(); i++) {
    Local<Object> jinc = Object::New(isolate);
    Local<Number> jradius = Number::New(isolate, incs.Radius());
    jinc->Set(String::NewFromUtf8(isolate, "radius"), jradius);
    Local<Number> jx = Number::New(isolate, incs.x(i));
    jinc->Set(String::NewFromUtf8(isolate, "x"), jx);
    Local<Number> jy = Number::New(isolate, incs.y(i));
    jinc->Set(String::NewFromUtf8(isolate, "y"), jy);
    Local<Number> jz = Number::New(isolate, incs.z(i));
    jinc->Set(String::NewFromUtf8(isolate, "z"), jz);
    jincs->Set(i, jinc);
  }
  jout->Set(String::NewFromUtf8(isolate, "inclusions"), jincs);

  args.GetReturnValue().Set(jout);
}

void LoadFrame(const FunctionCallbackInfo<Value>& args) {
  Isolate *isolate = args.GetIsolate();

  if (args.Length() < 2) {
    isolate->ThrowException(Exception::TypeError(
          String::NewFromUtf8(isolate, "Wrong number of arguments")));
    return;
  }

  if (!args[0]->IsString() || !args[1]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
          String::NewFromUtf8(isolate, "Wrong arguments")));
    return;
  }

  // input args
  String::Utf8Value dbname1(args[0]->ToString());
  std::string dbname(*dbname1);
  int frame = args[1]->NumberValue();

  std::vector<VortexLine> vlines;
  bool succ = LoadFrameFromDB(dbname, frame, vlines);

  // output 
  Local<Object> jout = Object::New(isolate);
  
  // vlines
  Local<Array> jvlines = Array::New(isolate);
  for (size_t i=0; i<vlines.size(); i++) {
    VortexLine& vline = vlines[i];
    vline.RemoveInvalidPoints();
    vline.Simplify(0.1);
    vline.ToBezier(0.01);
    vline.ToRegular(100);
    Local<Object> jvline = Object::New(isolate);

    // gid
    Local<Number> jgid = Number::New(isolate, vline.gid);
    jvline->Set(String::NewFromUtf8(isolate, "gid"), jgid);

    // verts
    Local<Array> jverts = Array::New(isolate);
    for (size_t j=0; j<vline.size(); j++)
      jverts->Set(j, Number::New(isolate, vline[j]));
    jvline->Set(String::NewFromUtf8(isolate, "verts"), jverts);
   
    // color
    // fprintf(stderr, "rgb=%d, %d, %d\n", vline.r, vline.g, vline.b);
    Local<Number> jr = Number::New(isolate, vline.r);
    jvline->Set(String::NewFromUtf8(isolate, "r"), jr);
    Local<Number> jg = Number::New(isolate, vline.g);
    jvline->Set(String::NewFromUtf8(isolate, "g"), jg);
    Local<Number> jb = Number::New(isolate, vline.b);
    jvline->Set(String::NewFromUtf8(isolate, "b"), jb);

    jvlines->Set(i, jvline);
  }
  jout->Set(String::NewFromUtf8(isolate, "vlines"), jvlines);
  
  args.GetReturnValue().Set(jout);
}

void Init(Local<Object> exports) {
  NODE_SET_METHOD(exports, "loadDataInfo", LoadDataInfo);
  NODE_SET_METHOD(exports, "loadFrame", LoadFrame);
}

NODE_MODULE(vf2, Init)
