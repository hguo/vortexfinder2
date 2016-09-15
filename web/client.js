var ws;

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
  var msg = JSON.parse(evt.data);
  console.log(msg);
  if (msg.type == "hdr") 
    updateHdr(msg.data);
  else if (msg.type == "vlines")
    updateVlines(msg.data);
}

function updateHdr(hdr) {
  console.log(hdr);
}

function updateVlines(vlines) {
  vortexId = [];
  vortexIdPos3D = [];
 
  for (i=0; i<vlines.length; i++) {
    var verts = vlines[i].verts;

    var r = vlines[i].r, g = vlines[i].g, b = vlines[i].b;
    var color = new THREE.Color(rgb(r, g, b));
  
    var points = [];
    for (j=0; j<verts.length/3; j++)
      points.push(new THREE.Vector3(verts[j*3], verts[j*3+1], verts[j*3+2]));
    var curve = new THREE.CatmullRomCurve3(points);

    var tubeGeometry = new THREE.TubeGeometry(curve, 100, 0.5, 8, false);
    var tubeMaterial = new THREE.MeshPhysicalMaterial({
      color: color,
      side: THREE.DoubleSide,
      wireframe: false
    });
    var tubeMesh = new THREE.Mesh(tubeGeometry, tubeMaterial);
    scene.add(tubeMesh);

    // var lineMaterial = new THREE.LineBasicMaterial({color: color});
    // var lineGeometry = new THREE.Geometry(curve);
    // var line = new THREE.Line(lineGeometry, lineMaterial);
    // scene.add(line);

    vortexId.push(vlines[i].gid);
    vortexIdPos3D.push(new THREE.Vector3(verts[0], verts[1], verts[2]));
  }

  render();
}

function onError(evt)
{
  console.log("error");
}

connectToServer();
