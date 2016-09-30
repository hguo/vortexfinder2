var stats = new Stats();
stats.showPanel(0);
stats.dom.id="stats";
document.body.appendChild(stats.dom);
$("#stats").css({visibility: "hidden"});

var clock = new THREE.Clock();
var scene = new THREE.Scene();

var camera = new THREE.PerspectiveCamera(30, window.innerWidth/window.innerHeight, 0.1, 1000); 
camera.position.z = 200;

var renderer = new THREE.WebGLRenderer({preserveDrawingBuffer: true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0xffffff, 1);
document.body.appendChild(renderer.domElement);

var pointLight = new THREE.PointLight(0xffffff);
pointLight.position.x = 100;
pointLight.position.y = 100;
pointLight.position.z = 100;
// pointLight.castShadow = true;
// pointLight.shadowDarkness = 0.5;
scene.add(pointLight);

var directionalLight = new THREE.DirectionalLight(0xffffff);
scene.add(directionalLight);

cameraControls = new THREE.TrackballControls(camera, renderer.domElement);
cameraControls.target.set(0, 0, 0);
cameraControls.zoomSpeed = 0.04;
cameraControls.panSpeed = 0.04;
// cameraControls.addEventListener("change", render); // not working.. sigh

var raycaster = new THREE.Raycaster();
var mousePos = new THREE.Vector2();

var dataCfg = {};
var dataHdrs = [];

var vortexCurves = [];
var vortexTubeMeshes = [];
var vortexLines = [];
var vortexColors = [];
var vortexColorsHex = [];
var vortexDistances = [];
var vortexEvents = [];
var inclusionSpheres = [];

var vortexId = [];
var vortexIdPos3D = [];
var displayVortexId = false;
var displayVortexTubes = true;
var vortexTubeRadius = 0.5;

window.addEventListener("mousedown", onMouseDown, false );
window.addEventListener("mousemove", onMouseMove, false);
window.addEventListener("resize", onResize, false);

function render() {
  stats.begin();

  raycaster.setFromCamera(mousePos, camera);
  var intersects = raycaster.intersectObjects(scene.children);
  // for (i=0; i<intersects.length; i++)
  //   intersects[i].object.material.color.set(0xff0000);

  // scene
  var delta = clock.getDelta();
  requestAnimationFrame(render);
  cameraControls.update(delta);
  directionalLight.position.copy(camera.position);
  renderer.render(scene, camera);

  if (displayVortexId)
    renderVortexId();

  stats.end();
}

function onMouseDown(evt) {
  mousePos.x = (evt.clientX / window.innerWidth) * 2 - 1;
  mousePos.y = -(evt.clientY / window.innerHeight) * 2 + 1;
}

function onMouseMove(evt) {
  mousePos.x = (evt.clientX / window.innerWidth) * 2 - 1;
  mousePos.y = -(evt.clientY / window.innerHeight) * 2 + 1;
}

function onResize() {
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  cameraControls.handleResize();
  
  $("#voltageChart").css({
    top: window.innerHeight - 120, // FIXME: hard code
    left: 0, 
    width: window.innerWidth, 
    height: 120, 
    position: "absolute"
  });
}

function toggleVortexId(on) {
  displayVortexId = on;
  if (!on) 
    clearVortexIdLabels();
}

function renderVortexId () {
  for (i=0; i<vortexIdPos3D.length; i++) {
    var vector = vortexIdPos3D[i].clone();
    vector.project(camera);
    vector.x = (vector.x + 1)/2 * window.innerWidth;
    vector.y = -(vector.y - 1)/2 * window.innerHeight;

    var text2 = document.getElementById("vortexId"+i);
    if (text2 == null) {
      text2 = document.createElement("div");
      text2.id = "vortexId" + i;
      text2.className = "vortexId";
      text2.style.position = "absolute";
      text2.style.fontSize = 12;
      document.body.appendChild(text2);
    }

    if (vector.x >= 0 && vector.x < window.innerWidth-50 && 
        vector.y >= 0 && vector.y < window.innerHeight-25 && 
        vector.z < 1) {
      text2.style.top = vector.y;
      text2.style.left = vector.x;
      text2.innerHTML = vortexId[i];
      text2.style.display = "block";
    } else {
      text2.style.display = "none";
    }
  }
}

function toggleTubes(on) {
  vortexTubeMeshes.forEach(function(tube) {
    tube.visible = on;
  });
}

function toggleLines(on) {
  vortexLines.forEach(function(line) {
    line.visible = on;
  });
}

function toggleInclusions(on)
{
  inclusionSpheres.forEach(function(sphere) {
    sphere.visible = on;
  });
}

function updateInclusions(incs)
{
  var sphereMaterial = new THREE.MeshPhongMaterial({
    color: 0xaaaaaa, 
    transparent: true,
    opacity: 0.9,
  });

  inclusionSpheres = [];
  for (i=0; i<incs.length; i++) {
    var sphereGeometry = new THREE.SphereGeometry(incs[i].radius, 50, 50, 0, Math.PI * 2, 0, Math.PI * 2);
    var sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.x = incs[i].x + dataCfg.Ox; // FIXME: just for Xfieldramp data
    sphere.position.y = incs[i].y + dataCfg.Oy; 
    sphere.position.z = incs[i].z + dataCfg.Oz;
    // sphere.castShadow = true;
    // sphere.receiveShadow = true;
    scene.add(sphere);
    inclusionSpheres.push(sphere);
  }
}

function updateVortexTubes(radius)
{
  vortexTubeMeshes.forEach(function(tube){scene.remove(tube);})
  vortexTubeMeshes = [];

  for (i=0; i<vortexCurves.length; i++) {
    var tubeGeometry = new THREE.TubeGeometry(vortexCurves[i], 100, radius, 8, false);
    var tubeMaterial = new THREE.MeshPhysicalMaterial({
      color: vortexColors[i],
      side: THREE.DoubleSide,
      wireframe: false
    });
    var tubeMesh = new THREE.Mesh(tubeGeometry, tubeMaterial);
    scene.add(tubeMesh);
    vortexTubeMeshes.push(tubeMesh);
  }
}

initializeControlPanel();
render();
