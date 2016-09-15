function connectToServer() {
  ws = new WebSocket("ws://127.0.0.1:8080");
  // ws.binaryType = "arraybuffer";
  ws.onopen = onOpen;
  ws.onclose = onClose;
  ws.onerror = onError;
  ws.onmessage = onMessage;
}

function onOpen(evt)
{
  console.log("connected to server");
  ws.send(JSON.stringify({
    dataname : "Xfieldramp", 
    frame : "1000"
  }));
}

function onClose(evt)
{
  console.log("connection closed");
}

function onMessage(evt)
{
  console.log("message");
  // var view = new DataView(evt);
}

function onError(evt)
{
  console.log("error");
}

connectToServer();
