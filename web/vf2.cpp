#include <node.h>
#include <rocksdb/db.h>
#include <string>
#include <sstream>
#include "common/VortexLine.h"

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

  std::stringstream ss;
  std::string buf;
  ss << "v." << frame;
  s = db->Get(rocksdb::ReadOptions(), ss.str(), &buf);
  // fprintf(stderr, "bufsize=%d\n", buf.size());

  diy::unserialize(buf, vlines);
  fprintf(stderr, "#vlines=%d\n", (int)vlines.size());

  delete db;
}

void Load(const FunctionCallbackInfo<Value>& args) {
  Isolate *isolate = args.GetIsolate();

  if (args.Length() < 6) {
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
  Local<Array> verts = Local<Array>::Cast(args[2]), 
               indices = Local<Array>::Cast(args[3]),
               colors = Local<Array>::Cast(args[4]),
               counts = Local<Array>::Cast(args[5]);

  int vertCount = 0;
  int vertCountSize = 0;
  for (int i=0; i<vlines.size(); i++) {
    const VortexLine& v = vlines[i];
    for (int j=0; j<v.size()/3; j++) {
      verts->Set(j*3, Number::New(isolate, v[j*3]));
      verts->Set(j*3+1, Number::New(isolate, v[j*3+1]));
      verts->Set(j*3+2, Number::New(isolate, v[j*3+2]));
      vertCount ++;
    }

    if (vertCount != 0) {
      counts->Set(vertCountSize ++, Number::New(isolate, vertCount));
      vertCount = 0;
    }
  }

  int cnt = 0;
  for (int i=0; i<vertCountSize; i++) {
    indices->Set(i, Number::New(isolate, i));
    cnt += counts->Get(i)->NumberValue();
  }
}

void Init(Local<Object> exports) {
  NODE_SET_METHOD(exports, "load", Load);
}

NODE_MODULE(vf2, Init)
