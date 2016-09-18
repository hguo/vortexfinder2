var ws;
var dbname = "GL_3D_Xfieldramp_inter_tet.rocksdb";
var currentFrame = 200;

function requestFrame(frame) {
  console.log("requesting frame " + frame + " in " + dbname);
  var msg = {
    type: "requestFrame",
    dbname: dbname,
    frame: currentFrame
  };

  if (ws.readyState == 1) ws.send(JSON.stringify(msg));
  else connectToServer();
}

function requestDataInfo() {
  console.log("requesting data info");
  var msg = {
    type: "requestDataInfo", 
    dbname: dbname
  };

  if (ws.readyState == 1) ws.send(JSON.stringify(msg));
  else connectToServer();
}

function clearVortexIdLabels() {
  $('.vortexId').remove();
}

function clearCurrentFrame() {
  console.log("cleanning current frame");
  vortexTubeMeshes.forEach(function(tube) {scene.remove(tube);});
  vortexLines.forEach(function(line) {scene.remove(line);});
  clearVortexIdLabels();

  vortexCurves = [];
  vortexLines = [];
  vortexTubeMeshes = [];
  vortexColors = [];
  vortexColorsHex = [];
  vortexId = [];
  vortexIdPos3D = [];
  vortexDistances = [];
}

function connectToServer() {
  clearCurrentFrame();

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
  requestDataInfo();
  requestFrame(currentFrame);
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
  if (msg.type == "dataInfo") 
    updateDataInfo(msg.data);
  else if (msg.type == "vlines") {
    updateVlines(msg.data.vlines);
    updateDistances(msg.data.dist);
  }
}

function updateDataInfo(info) {
  dataCfg = info.cfg;
  dataHdrs = info.hdrs;
  updateInclusions(info.inclusions);

  // dataHdrs.forEach(function(d) {if (d.V<0) d.V=0;});
  createLineChart();
}

function updateFrameInfo() {
  var hdr = dataHdrs[currentFrame];
  var frameinfo = document.getElementById("frameinfo");
  if (frameinfo == null) {
    frameinfo = document.createElement("div");
    frameinfo.id = "frameinfo";
    frameinfo.style.position = "absolute";
    frameinfo.style.top = 15;
    frameinfo.style.left = 15;
    frameinfo.style.fontSize = 20;
    document.body.appendChild(frameinfo);
  }
  frameinfo.innerHTML = 
    "frame=" + currentFrame + ", " +
    "timestep=" + hdr.timestep + ", " + 
    "t=" + (hdr.timestep * dataCfg.dt).toFixed(3) + ", " +
    "B=(" + hdr.Bx.toFixed(3) + ", " + hdr.By.toFixed(3) + ", " + hdr.Bz.toFixed(3) + "), " +
    "V=" + hdr.V.toFixed(3);

  var frameCursor = d3.select("#frameCursor")
    .attr("transform", "translate(" + xScale(dataHdrs[currentFrame].timestep*dataCfg.dt) + ", 0)");
}

function rgb2hex(r, g, b) {
  var hex = b | (g<<8) | (r<<16);
  return "#" + hex.toString(16);
}

function updateDistances(dist) {
  var nv = vortexId.length;
  vortexDistances = [];
  for (i=0; i<nv; i++) {
    for (j=0; j<nv; j++) {
      if (i==j) continue;
      vortexDistances.push({
        source: vortexId[i],
        target: vortexId[j],
        dist: dist[i*nv+j]
      });
      if (vortexId[i] == 133 && vortexId[j] == 191) console.log(dist[i*nv+j]);
      if (vortexId[i] == 194 && vortexId[j] == 195) console.log(dist[i*nv+j]);
    }
  }

  newMdsNodes = [];
  vortexId.forEach(function(v, i) {
    var node = mdsNodes.find(function(d) {return d.id == v;});
    if (node == undefined) {
      node = {
        id: v,
        vx: 0,
        vy: 0,
        x: 160, // FIXME: hardcode
        y: 160,
        color: vortexColorsHex[i]
      }
    }
      newMdsNodes.push(node);
  });

  mdsNodes = newMdsNodes;

  updateMDSChart();
}

function updateVlines(vlines) {
  clearCurrentFrame();

  for (i=0; i<vlines.length; i++) {
    var verts = vlines[i].verts;
    // console.log(vlines[i].moving_speed);

    var r = vlines[i].r, g = vlines[i].g, b = vlines[i].b;
    var color = new THREE.Color(rgb(r, g, b));
    vortexColors.push(color);
    vortexColorsHex.push(rgb2hex(r, g, b));
   
    var lineGeometry = new THREE.Geometry();
    var points = [];
    for (j=0; j<verts.length/3; j++) {
      points.push(new THREE.Vector3(verts[j*3], verts[j*3+1], verts[j*3+2]));
      lineGeometry.vertices.push(new THREE.Vector3(verts[j*3], verts[j*3+1], verts[j*3+2]));
    }
    var curve = new THREE.CatmullRomCurve3(points);
    vortexCurves.push(curve);

    var lineMaterial = new THREE.LineBasicMaterial({color: color});
    var line = new THREE.Line(lineGeometry, lineMaterial);
    vortexLines.push(line);
    scene.add(line);

    vortexId.push(vlines[i].gid);
    vortexIdPos3D.push(new THREE.Vector3(verts[0], verts[1], verts[2]));
  }

  updateVortexTubes(vortexTubeRadius);
  if (displayVortexTubes) {
    toggleTubes(true); toggleLines(false);
  } else {
    toggleTubes(false); toggleLines(true);
  }

  updateFrameInfo();
}

function onError(evt)
{
  console.log("error");
}

connectToServer();
