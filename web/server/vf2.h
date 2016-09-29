#include <node.h>
#include <node_object_wrap.h>
#include <rocksdb/db.h>
#include "common/VortexLine.h"
#include "common/VortexTransition.h"
#include "common/Inclusions.h"
  
using namespace v8;

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

class VF2 : public node::ObjectWrap {
public:
  static void Init(Local<Object> exports);

private:
  explicit VF2() : db(NULL) {}
  ~VF2() {CloseDB();}

  static void New(const FunctionCallbackInfo<Value>& args);
  static void OpenDB(const FunctionCallbackInfo<Value>& args);
  static void GetDataInfo(const FunctionCallbackInfo<Value>& args);
  static void GetEvents(const FunctionCallbackInfo<Value>& args);
  static void LoadFrame(const FunctionCallbackInfo<Value>& args);

  static Persistent<Function> constructor;

private:
  bool OpenDB(const std::string& dbname);
  void CloseDB();
  void LoadDataInfo();
  bool LoadFrame(int frame,
      std::vector<VortexLine>& vlines,
      std::vector<float>& dist);

private:
  std::string dbname;
  rocksdb::DB* db;

  vfgpu_cfg_t cfg;
  std::vector<vfgpu_hdr_t> hdrs;
  Inclusions incs;
  VortexTransition vt;
};
