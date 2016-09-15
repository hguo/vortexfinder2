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
  // ws.send(JSON.stringify({
  //   dataname : "Xfieldramp", 
  //   frame : "1000"
  // }));
}

function onClose(evt)
{
  console.log("connection closed");
}

function rgb(r, g, b) {
  return "rgb("+r+","+g+","+b+")";
}

function onMessage(evt)
{
  vlines = JSON.parse(evt.data);
  // console.log(vlines);
 
  for (i=0; i<vlines.length; i++) {
    var verts = vlines[i].verts;
    var geom = new THREE.Geometry();
    for (j=0; j<verts.length/3; j++)
      geom.vertices.push(new THREE.Vector3(verts[j*3], verts[j*3+1], verts[j*3+2]));

    var r = vlines[i].r, g = vlines[i].g, b = vlines[i].b;
    var color = new THREE.Color(rgb(r, g, b));
  
    // var tubeGeom = new THREE.TubeGeometry(geom, 20, 2, 8, false);
    // var tube = new THREE.Mesh(tubeGeom);

    var material = new THREE.LineBasicMaterial({color: color});
    var line = new THREE.Line(geom, material);
    scene.add(line);
  }

  render();
}

function onError(evt)
{
  console.log("error");
}

connectToServer();
