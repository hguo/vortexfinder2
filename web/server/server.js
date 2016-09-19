var WebSocketServer = require("ws").Server;
var glob = require("glob");
var vf2 = require('bindings')('vf2');

// var bson = require("bson");
// var BSON = new bson.BSONPure.BSON()
// var BSON = new bson.BSONPure.BSON()

var wss = new WebSocketServer({
  port : 8080, 
  // binaryType : "arraybuffer",
  perMessageDeflate : "false"
});

wss.on("connection", function(ws) {
  console.log("connected.");
  sendDBList(ws);

  ws.on("message", function(data) {
    var msg = JSON.parse(data);
    if (msg.type == "requestDataInfo") {
      sendDataInfo(ws, msg.dbname);
    } else if (msg.type == "requestFrame") {
      sendFrame(ws, msg.dbname, msg.frame);
    }
  });

  ws.on("close", function() {
    console.log("disconnected.");
  });
})

wss.on("close", function(ws) {
  console.log("closed.");
})

function sendDBList(ws) {
  glob("*rocksdb", function(er, files) {
    msg = {
      type: "dbList", 
      data: files
     };
    ws.send(JSON.stringify(msg));
  })
}

function sendDataInfo(ws, dbname) {
  console.log("requested data info");
  var dataInfo = vf2.loadDataInfo(dbname);
  
  msg = {
    type: "dataInfo", 
    data: dataInfo
   };
  ws.send(JSON.stringify(msg));
}

function sendFrame(ws, dbname, frame) {
  console.log("requested frame " + frame + " in " + dbname);

  var frameData = vf2.loadFrame(dbname, frame);
 
  msg = {
    type: "vlines", 
    data: frameData
  };
  ws.send(JSON.stringify(msg));

  // ws.send(BSON.serialize(vlines)); // I don't know why BSON doesn't work

  // ws.onmessage = function(msg) {
  //   console.log(msg.data);
  // }
}
