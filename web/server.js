var WebSocketServer = require("ws").Server;
var vf2 = require('bindings')('vf2');

// var bson = require("bson");
// var BSON = new bson.BSONPure.BSON()
// var BSON = new bson.BSONPure.BSON()

const dbname="/Users/hguo/workspace/projects/vortexfinder2/build/bin/GL_3D_Xfieldramp_inter.rocksdb";

var verts = [];
var indices = [];
var colors = [];
var counts = [];

wss = new WebSocketServer({
  port : 8080, 
  // binaryType : "arraybuffer",
  perMessageDeflate : "false"
});

wss.on("connection", function(ws) {
  console.log("connected.");

  vf2.load(dbname, 100, verts, indices, colors, counts);

  // ws.send(BSON.serialize(verts)); // I don't know why BSON doesn't work

  var data = [verts, indices, colors, counts];
  ws.send(JSON.stringify(data));

  // ws.onmessage = function(msg) {
  //   console.log(msg.data);
  // }
})
  
