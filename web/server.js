var WebSocketServer = require("ws").Server;
var vf2 = require('bindings')('vf2');

// var bson = require("bson");
// var BSON = new bson.BSONPure.BSON()
// var BSON = new bson.BSONPure.BSON()

const dbname="GL_3D_Xfieldramp_inter.rocksdb";

wss = new WebSocketServer({
  port : 8080, 
  // binaryType : "arraybuffer",
  perMessageDeflate : "false"
});

wss.on("connection", function(ws) {
  console.log("connected.");

  var vlines = [];
  var hdr = {};

  vf2.load(dbname, 200, hdr, vlines);
  console.log(hdr);
 
  msg0 = {
    type: "vlines", 
    data: vlines
  };
  ws.send(JSON.stringify(msg0));

  msg1 = {
    type: "hdr", 
    data: hdr
  };
  ws.send(JSON.stringify(msg1));

  // ws.send(BSON.serialize(vlines)); // I don't know why BSON doesn't work

  // ws.onmessage = function(msg) {
  //   console.log(msg.data);
  // }
})

wss.on("close", function(ws) {
  console.log("closed.");
})
