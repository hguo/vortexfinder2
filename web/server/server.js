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

  var obj = new vf2.vf2();

  ws.on("message", function(data) {
    var msg = JSON.parse(data);
    if (msg.type == "requestDataInfo") {
      sendDataInfo(ws, obj, msg.dbname);
    } else if (msg.type == "requestFrame") {
      sendFrame(ws, obj, msg.frame);
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

function sendDataInfo(ws, obj, dbname) {
  console.log("requested data info");
  obj.openDB(dbname);
  var dataInfo = obj.getDataInfo();
  var events = obj.getEvents();

  msg = {
    type: "dataInfo", 
    dataInfo: dataInfo, 
    events: events
  };
  ws.send(JSON.stringify(msg));
}

function sendFrame(ws, obj, frame) {
  console.log("requested frame " + frame);
  var frameData = obj.loadFrame(frame);
 
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
