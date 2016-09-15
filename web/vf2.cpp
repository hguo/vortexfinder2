#include <node.h>
#include <rocksdb/db.h>
#include <string>
#include <sstream>
#include "common/VortexLine.h"
#include "common/VortexTransition.h"

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

void LoadVorticiesFromDB(const std::string& dbname, int frame, std::vector<VortexLine>& vlines)
{
  fprintf(stderr, "dbname=%s, frame=%d\n", 
      dbname.c_str(), frame);

  rocksdb::DB* db;
  rocksdb::Options options;
  rocksdb::Status s;
  
  s = rocksdb::DB::Open(options, dbname, &db);
  assert(s.ok());

  std::string buf;
  s = db->Get(rocksdb::ReadOptions(), "trans", &buf);
  VortexTransition vt;
  diy::unserialize(buf, vt);

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
  LoadVorticiesFromDB(dbname, frame, vlines);

  // output args
  Local<Array> jvlines = Local<Array>::Cast(args[2]);
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
}

NODE_MODULE(vf2, Init)
