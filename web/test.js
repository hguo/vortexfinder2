var glob = require("glob");

glob("*rocksdb", function(er, files) {
  console.log(files);
})

// const vf2 = require('bindings')('vf2');
// const dbname="/Users/hguo/workspace/projects/vortexfinder2/build/bin/GL_3D_Xfieldramp_inter.rocksdb";

// var vlines = [];
// vf2.load(dbname, 100, vlines);
// console.log(vlines);
