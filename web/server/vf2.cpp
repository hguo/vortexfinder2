#include "vf2.h"
#include <string>
#include <sstream>

Persistent<Function> VF2::constructor;

void VF2::Init(Local<Object> exports) {
  Isolate *isolate = exports->GetIsolate();
  Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, New);
  tpl->SetClassName(String::NewFromUtf8(isolate, "vf2"));
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  NODE_SET_PROTOTYPE_METHOD(tpl, "openDB", OpenDB);
  NODE_SET_PROTOTYPE_METHOD(tpl, "getDataInfo", GetDataInfo);
  NODE_SET_PROTOTYPE_METHOD(tpl, "getEvents", GetEvents);
  NODE_SET_PROTOTYPE_METHOD(tpl, "loadFrame", LoadFrame);

  constructor.Reset(isolate, tpl->GetFunction());
  exports->Set(String::NewFromUtf8(isolate, "vf2"), 
      tpl->GetFunction());
}

void VF2::New(const FunctionCallbackInfo<Value>& args) {
  Isolate *isolate = args.GetIsolate();

  if (args.IsConstructCall()) {
    VF2 *obj = new VF2;
    obj->Wrap(args.This());
    args.GetReturnValue().Set(args.This());
  } else {
    const int argc = 1;
    Local<Value> argv[argc] = {args[0]}; 
    Local<Context> context = isolate->GetCurrentContext();
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> result = 
      cons->NewInstance(context, argc, argv).ToLocalChecked();
    args.GetReturnValue().Set(result);
  }
}

void VF2::OpenDB(const FunctionCallbackInfo<Value>& args) {
  Isolate *isolate = args.GetIsolate();
  VF2* obj = ObjectWrap::Unwrap<VF2>(args.Holder());

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
  
  String::Utf8Value dbname1(args[0]->ToString());
  std::string dbname(*dbname1);

  obj->OpenDB(dbname);
}

void VF2::GetEvents(const FunctionCallbackInfo<Value>& args) {
  Isolate *isolate = args.GetIsolate();
  VF2* obj = ObjectWrap::Unwrap<VF2>(args.Holder());
  const VortexTransition& vt = obj->vt;
  const std::vector<VortexEvent>& events = vt.Events();

  Local<Array> jevents = Array::New(isolate);
  for (int i=0; i<events.size(); i++) {
    const VortexEvent& e = events[i];
    Local<Object> jevent = Object::New(isolate);
    
    Local<Number> jf0 = Number::New(isolate, e.if0);
    jevent->Set(String::NewFromUtf8(isolate, "f0"), jf0);
    
    Local<Number> jf1 = Number::New(isolate, e.if1);
    jevent->Set(String::NewFromUtf8(isolate, "f1"), jf1);

    Local<Number> jtype = Number::New(isolate, e.type);
    jevent->Set(String::NewFromUtf8(isolate, "type"), jtype);

    Local<Array> jlhs = Array::New(isolate);
    int k = 0;
    for (std::set<int>::iterator it = e.lhs.begin(); it != e.lhs.end(); it ++) 
      jlhs->Set(Number::New(isolate, k ++), Number::New(isolate, *it));
    jevent->Set(String::NewFromUtf8(isolate, "lhs"), jlhs);

    Local<Array> jrhs = Array::New(isolate);
    k = 0;
    for (std::set<int>::iterator it = e.rhs.begin(); it != e.rhs.end(); it ++) 
      jrhs->Set(Number::New(isolate, k ++), Number::New(isolate, *it));
    jevent->Set(String::NewFromUtf8(isolate, "rhs"), jrhs);

    jevents->Set(Number::New(isolate, i), jevent);
  }

  args.GetReturnValue().Set(jevents);
}

void VF2::GetDataInfo(const FunctionCallbackInfo<Value>& args) {
  Isolate *isolate = args.GetIsolate();
  VF2* obj = ObjectWrap::Unwrap<VF2>(args.Holder());
  
  obj->LoadDataInfo();
  const vfgpu_cfg_t& cfg = obj->cfg;
  const std::vector<vfgpu_hdr_t> &hdrs = obj->hdrs;
  const Inclusions& incs = obj->incs;

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

void VF2::LoadFrame(const FunctionCallbackInfo<Value>& args) {
  Isolate *isolate = args.GetIsolate();

  VF2* obj = ObjectWrap::Unwrap<VF2>(args.Holder());
  // const std::string& dbname = obj->dbname;

  if (args.Length() < 1) {
    isolate->ThrowException(Exception::TypeError(
          String::NewFromUtf8(isolate, "Wrong number of arguments")));
    return;
  }

  if (!args[0]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
          String::NewFromUtf8(isolate, "Wrong arguments")));
    return;
  }

  // input args
  const int frame = args[0]->NumberValue();

  std::vector<VortexLine> vlines;
  std::vector<float> dist;
  bool succ = obj->LoadFrame(frame, vlines, dist);

  // output 
  Local<Object> jout = Object::New(isolate);
  
  // vlines
  Local<Array> jvlines = Array::New(isolate);
  for (size_t i=0; i<vlines.size(); i++) {
    VortexLine& vline = vlines[i];
    if (vline.is_bezier) {
      vline.ToRegular(100);
    } else {
      vline.RemoveInvalidPoints();
      vline.Simplify(0.1);
      // vline.ToBezier(0.01);
      // vline.ToRegular(100);
    }
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

    // moving_speed
    Local<Number> jms = Number::New(isolate, vline.moving_speed);
    jvline->Set(String::NewFromUtf8(isolate, "moving_speed"), jms);

    jvlines->Set(i, jvline);
  }
  jout->Set(String::NewFromUtf8(isolate, "vlines"), jvlines);
 
  // distance matrix
  Local<Array> jdist = Array::New(isolate);
  for (size_t i=0; i<dist.size(); i++) 
    jdist->Set(i, Number::New(isolate, dist[i]));
  jout->Set(String::NewFromUtf8(isolate, "dist"), jdist);

  args.GetReturnValue().Set(jout);
}

bool VF2::OpenDB(const std::string& dbname_)
{
  dbname = dbname_;
  rocksdb::Options options;
  options.create_if_missing = false;

  rocksdb::Status s = rocksdb::DB::OpenForReadOnly(options, dbname, &db);
  fprintf(stderr, "Openning db, dbname=%s, succ=%d\n", dbname.c_str(), (bool)s.ok());

  if (s.ok()) {
    LoadDataInfo();
    return true;
  } else return false;
}

void VF2::CloseDB()
{
  if (db != NULL) {
    dbname.clear();
    delete db;
    db = NULL;
  }
}

void VF2::LoadDataInfo()
{
  rocksdb::Status s;
  std::string buf;

  s = db->Get(rocksdb::ReadOptions(), "cfg", &buf);
  if (buf.size() > 0) 
    diy::unserialize(buf, cfg);
  
  s = db->Get(rocksdb::ReadOptions(), "hdrs", &buf);
  if (buf.size() > 0) 
    diy::unserialize(buf, hdrs);

  s = db->Get(rocksdb::ReadOptions(), "inclusions", &buf);
  if (buf.size() > 0) 
    diy::unserialize(buf, incs);
  
  s = db->Get(rocksdb::ReadOptions(), "trans", &buf);
  if (buf.size() > 0) {
    diy::unserialize(buf, vt);
    srand(100);
    vt.SequenceGraphColoring(); // TODO
  }
}

bool VF2::LoadFrame(
    int frame, 
    std::vector<VortexLine>& vlines,
    std::vector<float>& dist)
{
  fprintf(stderr, "dbname=%s, frame=%d\n", dbname.c_str(), frame);

  rocksdb::Status s;
  std::string buf;

  const int timestep = vt.Frame(frame);
  std::stringstream ss;
  ss << "v." << timestep;
  buf.clear();
  s = db->Get(rocksdb::ReadOptions(), ss.str(), &buf);
  if (buf.empty()) return false;
  diy::unserialize(buf, vlines);

  for (size_t i=0; i<vlines.size(); i++) {
    vlines[i].gid = vt.lvid2gvid(frame, vlines[i].id); // sorry, this is confusing
    vt.SequenceColor(vlines[i].gid, vlines[i].r, vlines[i].g, vlines[i].b);
  }

  // distance matrix
  ss.clear();
  ss << "d." << timestep;
  buf.clear();
  s = db->Get(rocksdb::ReadOptions(), ss.str(), &buf);
  if (!buf.empty())
    diy::unserialize(buf, dist);

  return true;
}

void Init(Local<Object> exports) {
  VF2::Init(exports);
}

NODE_MODULE(vf2, Init);
