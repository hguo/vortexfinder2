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

  ws.on("message", function(data) {
    var msg = JSON.parse(data);
    if (msg.type == "requestDataInfo") {
      sendDataInfo(ws, msg.dbname);
    } else if (msg.type == "requestFrame") {
      sendFrame(ws, msg.dbname, msg.frame);
    }
  });
})

wss.on("close", function(ws) {
  console.log("closed.");
})

function sendDataInfo(ws, dbname) {
  console.log("requested data info");
  var inclusions = vf2.loadInclusions(dbname);
  
  msg2 = {
    type: "inclusions", 
    data: inclusions
   };
  ws.send(JSON.stringify(msg2));
}

function sendFrame(ws, dbname, frame) {
  console.log("requested frame " + frame + " in " + dbname);

  var vlines = [];
  var hdr = {};

  vf2.load(dbname, frame, hdr, vlines);
 
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
}