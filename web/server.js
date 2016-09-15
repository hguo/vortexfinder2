var WebSocketServer = require("ws").Server;

wss = new WebSocketServer({
  port : 8080, 
  // binaryType : "arraybuffer",
  perMessageDeflate : "false"
});

wss.on("connection", function(ws) {
  console.log("connected.");

  ws.send("hello");

  ws.onmessage = function(msg) {
    console.log(msg.data);
  }
  // ws.on("message", function(msg) {
  //   console.log("message.");
  // })
})
  
