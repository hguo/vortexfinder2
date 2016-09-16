var ws;
var dbname = "GL_3D_Xfieldramp_inter.rocksdb";
var currentFrame = 200;

function requestCurrentFrame() {
  console.log("requesting frame " + currentFrame + " in " + dbname);
  var msg = {
    type: "requestFrame",
    dbname: dbname,
    frame: currentFrame
  };

  ws.send(JSON.stringify(msg));
}

function clearCurrentFrame() {
  console.log("cleanning current frame");
  vortexTubes.forEach(function(tube) {scene.remove(tube);})
  inclusionSpheres.forEach(function(sphere) {scene.remove(sphere);})

  vortexIdLabels = document.getElementsByClassName("vortexId");
  for (i=0; i<vortexIdLabels.length; i++) 
    document.body.removeChild(vortexIdLabels[i]);

  vortexCurves = [];
  vortexTubes = [];
  vortexColors = [];
  vortexId = [];
  vortexIdPos3D = [];
}

function connectToServer() {
  // ws = new WebSocket("ws://red.mcs.anl.gov:8080");
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
  requestCurrentFrame();
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
  // console.log(msg);
  if (msg.type == "hdr") 
    updateHdr(msg.data);
  else if (msg.type == "vlines")
    updateVlines(msg.data);
  else if (msg.type == "inclusions")
    updateInclusions(msg.data);
}

function updateHdr(hdr) {
  // console.log(hdr);
  var datainfo = document.getElementById("datainfo");
  if (datainfo == null) {
    datainfo = document.createElement("div");
    datainfo.id = "datainfo";
    datainfo.style.position = "absolute";
    datainfo.style.top = 15;
    datainfo.style.left = 15;
    datainfo.style.fontSize = 20;
    document.body.appendChild(datainfo);
  }
  datainfo.innerHTML = 
    "timestep=" + hdr.timestep + ", " + 
    "B=(" + hdr.Bx.toFixed(3) + ", " + hdr.By.toFixed(3) + ", " + hdr.Bz.toFixed(3) + "), " +
    "V=" + hdr.V.toFixed(3);
}

function updateVlines(vlines) {
  clearCurrentFrame();

  for (i=0; i<vlines.length; i++) {
    var verts = vlines[i].verts;

    var r = vlines[i].r, g = vlines[i].g, b = vlines[i].b;
    var color = new THREE.Color(rgb(r, g, b));
    vortexColors.push(color);
  
    var points = [];
    for (j=0; j<verts.length/3; j++)
      points.push(new THREE.Vector3(verts[j*3], verts[j*3+1], verts[j*3+2]));
    var curve = new THREE.CatmullRomCurve3(points);
    vortexCurves.push(curve);

    // var lineMaterial = new THREE.LineBasicMaterial({color: color});
    // var lineGeometry = new THREE.Geometry(curve);
    // var line = new THREE.Line(lineGeometry, lineMaterial);
    // scene.add(line);

    vortexId.push(vlines[i].gid);
    vortexIdPos3D.push(new THREE.Vector3(verts[0], verts[1], verts[2]));
  }

  updateVortexTubes(0.5);

  render();
}

function onError(evt)
{
  console.log("error");
}

connectToServer();
