#/bin/sh

export LD_LIBRARY_PATH=/nfs/proj-tpeterka/hguo/compile/rocksdb-4.9:/nfs/proj-tpeterka/hguo/projects/vortexfinder2/build/install/lib:$LD_LIBRARY_PATH

/nfs/proj-tpeterka/hguo/local.gcc-5.1.0/node-4.5.0/bin/node server.js
