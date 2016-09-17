var stats = new Stats();
stats.showPanel(1);
document.body.appendChild(stats.dom);

var clock = new THREE.Clock();
var scene = new THREE.Scene();

var camera = new THREE.PerspectiveCamera(30, window.innerWidth/window.innerHeight, 0.1, 1000); 
camera.position.z = 200;

var renderer = new THREE.WebGLRenderer();
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
cameraControls.panSpeed = 0.8;
// cameraControls.addEventListener("change", render); // not working.. sigh

var dataCfg = {};
var dataHdrs = [];

var vortexCurves = [];
var vortexTubes = [];
var vortexColors = [];
var inclusionSpheres = [];

var vortexId = [];
var vortexIdPos3D = [];
var displayVortexId = false;

window.addEventListener("resize", onResize, false);

function render() {
  stats.begin();

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

function onResize() {
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  cameraControls.handleResize();

  $("svg").css({
    top: window.innerHeight - 120, 
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
      text2.style.fontSize = 10;
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
  vortexTubes.forEach(function(tube){scene.remove(tube);})
  vortexTubes = [];

  for (i=0; i<vortexCurves.length; i++) {
    var tubeGeometry = new THREE.TubeGeometry(vortexCurves[i], 100, radius, 8, false);
    var tubeMaterial = new THREE.MeshPhysicalMaterial({
      color: vortexColors[i],
      side: THREE.DoubleSide,
      wireframe: false
    });
    var tubeMesh = new THREE.Mesh(tubeGeometry, tubeMaterial);
    scene.add(tubeMesh);
    vortexTubes.push(tubeMesh);
  }
}

render();
