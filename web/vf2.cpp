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

void LoadInclusionsFromDB(const std::string& dbname, Inclusions& incs)
{
  rocksdb::DB* db;
  rocksdb::Status s;
  
  rocksdb::Options options;
  options.create_if_missing = false;
  s = rocksdb::DB::Open(options, dbname, &db);
  assert(s.ok());
  
  std::string buf;
  s = db->Get(rocksdb::ReadOptions(), "inclusions", &buf);
  diy::unserialize(buf, incs);

  delete db;
}

void LoadVorticiesFromDB(const std::string& dbname, int frame, vfgpu_hdr_t& hdr, std::vector<VortexLine>& vlines)
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
  VortexTransition vt;
  diy::unserialize(buf, vt);
 
  std::vector<vfgpu_hdr_t> hdrs;
  s = db->Get(rocksdb::ReadOptions(), "f", &buf);
  diy::unserialize(buf, hdrs);
  hdr = hdrs[frame];

  const int timestep = vt.Frame(frame);
  std::stringstream ss;
  ss << "v." << timestep;
  buf.clear();
  s = db->Get(rocksdb::ReadOptions(), ss.str(), &buf);
  diy::unserialize(buf, vlines);

  for (size_t i=0; i<vlines.size(); i++) {
    vlines[i].gid = vt.lvid2gvid(frame, vlines[i].id); // sorry, this is confusing
    vt.SequenceColor(vlines[i].gid, vlines[i].r, vlines[i].g, vlines[i].b);
    // fprintf(stderr, "timestep=%d, lid=%d, gid=%d\n", timestep, vlines[i].id, vlines[i].gid);
  }
  // fprintf(stderr, "#vlines=%d\n", (int)vlines.size());

  delete db;
}

void LoadInclusions(const FunctionCallbackInfo<Value>& args) {
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

  Inclusions incs;
  LoadInclusionsFromDB(dbname, incs);

  // outputs
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

  args.GetReturnValue().Set(jincs);
}

void Load(const FunctionCallbackInfo<Value>& args) {
  Isolate *isolate = args.GetIsolate();

  if (args.Length() < 3) {
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
  vfgpu_hdr_t hdr;
  LoadVorticiesFromDB(dbname, frame, hdr, vlines);

  // output args
  Local<Object> jhdr = Local<Object>::Cast(args[2]);
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
  
  // vlines
  Local<Array> jvlines = Local<Array>::Cast(args[3]);
  for (size_t i=0; i<vlines.size(); i++) {
    VortexLine& vline = vlines[i];
    vline.ToBezier();
    vline.ToRegular(0.2);
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
}

void Init(Local<Object> exports) {
  NODE_SET_METHOD(exports, "load", Load);
  NODE_SET_METHOD(exports, "loadInclusions", LoadInclusions);
}

NODE_MODULE(vf2, Init)
