const vf2 = require('bindings')('vf2');
const dbname="/Users/hguo/workspace/projects/vortexfinder2/build/bin/GL_3D_Xfieldramp_inter.rocksdb";

var verts = [];
var indices = [];
var colors = [];
var counts = [];

vf2.load(dbname, 100, verts, indices, colors, counts);
// console.log(counts);
