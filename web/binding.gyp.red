{
  "targets": [
    {
      "target_name": "vf2", 
      "sources": [ "vf2.cpp" ],
      "include_dirs": [
        "/nfs/proj-tpeterka/hguo/compile/rocksdb-4.9/include",
        "/nfs/proj-tpeterka/hguo/projects/vortexfinder2/build/install/include",
        "/nfs/proj-tpeterka/hguo/projects/vortexfinder2/diy2/include"
      ],
      "libraries": [
        "/nfs/proj-tpeterka/hguo/projects/vortexfinder2/build/install/lib/libglcommon.so",
        "/nfs/proj-tpeterka/hguo/compile/rocksdb-4.9/librocksdb.so.4.9.0"
      ],
      "xcode_settings": {
        "OTHER_CPLUSPLUSFLAGS" : [ "-std=c++11", "-stdlib=libc++" ], 
        "OTHER_LDFLAGS": [ "-stdlib=libc++" ],
        "MACOSX_DEPLOYMENT_TARGET": "10.7",
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES"
      }, 
    }
  ]
}
