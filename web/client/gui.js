var menuText = function() {
  this.dataName = "";
  this.frame = 200;
  this.resetTrackball = function () {
    cameraControls.reset();
  };
  this.displayStats = false;
  this.displayVoltage = true;
  this.displayVortexId = false;
  this.displayInclusions = true;
  this.displayMDS = false;
  this.displayEvents = false;
  this.distScale = 2;
  this.vortexRendering = "tube";
  this.tubeRadius = 0.5;
  this.nextFrame = function() {
    currentFrame ++;
    requestFrame(currentFrame);
  };
  this.previousFrame = function() {
    currentFrame --;
    requestFrame(currentFrame);
  };

  this.saveImage = function() {
    window.open( renderer.domElement.toDataURL( 'image/png' ), 'screenshot' );
  };

  this.saveTrackball = function () {
    var trac = {
      target: cameraControls.target,
      position: cameraControls.object.position,
      up: cameraControls.object.up
    };
    var blob = new Blob([JSON.stringify(trac)], {type: "text/plain;charset=utf-8"});
    saveAs(blob, "camera.json");
  };

  this.loadTrackball = function () {
    if (!window.File || !window.FileReader || !window.FileList || !window.Blob) {
      alert('The File APIs are not fully supported in this browser.');
      return;
    }

    var evt = document.createEvent("MouseEvents");
    evt.initEvent("click", true, false);
    file_open.dispatchEvent(evt);
    file_open.onchange = function() {
      var path = file_open.value;
      var f = file_open.files[0];
      var reader = new FileReader();
      
      reader.onload = function(evt) {
        var str = evt.target.result;
        var trac = JSON.parse(str);
        cameraControls.load(trac.target, trac.position, trac.up);
      }

      reader.readAsText(f);
    }
  };
};

function initializeControlPanel () {
  var text = new menuText();
  var gui = new dat.GUI();

  var f1 = gui.addFolder("Data");
  // f1.add(text, 'dataName');
  // f1.add(text, 'frame').onChange(function(val) {
  //   currentFrame = val;
  //   requestFrame();
  // });
  f1.add(text, 'previousFrame');
  f1.add(text, 'nextFrame');
  f1.open();

  var f2 = gui.addFolder("3D Rendering");
  f2.add(text, "displayStats").onChange(function(val) {
    if (val) $("#stats").css({visibility: "visible"});
    else $("#stats").css({visibility: "hidden"});
  });
  f2.add(text, 'displayVortexId').onChange(function(val) {
    toggleVortexId(val);
  });
  f2.add(text, "displayInclusions").onChange(function(on) {
    toggleInclusions(on);
  });
  f2.add(text, "vortexRendering", ["tube", "line"]).onChange(function(val) {
    if (val == "tube") {
      displayVortexTubes = true;
      toggleTubes(true); 
      toggleLines(false);
    } else {
      displayVortexTubes = false;
      toggleTubes(false);
      toggleLines(true);
    }
  });
  f2.add(text, "tubeRadius", 0.1, 2).onChange(function(val) {
    vortexTubeRadius = val;
    updateVortexTubes(val);
  });
  f2.add(text, "saveImage");
  f2.open();

  var f3 = gui.addFolder("Trackball");
  f3.add(text, "saveTrackball");
  f3.add(text, "loadTrackball");
  f3.add(text, "resetTrackball");

  var f4 = gui.addFolder("Charts");
  f4.add(text, "displayVoltage").onChange(function(val) {
    if (val) $("#voltageChart").css({visibility: "visible"});
    else $("#voltageChart").css({visibility: "hidden"});
  });
  f4.add(text, "displayMDS").onChange(function(val) {
    displayMDS = val;
    if (val) {
      updateMDSChart();
      $("#mdsChart").css({visibility: "visible"});
    } else {
      $("#mdsChart").css({visibility: "hidden"});
    }
  });
  f4.add(text, "displayEvents").onChange(function(val) {
    if (val) $("#eventCursors").css({display: "block"});
    else $("#eventCursors").css({display: "none"});
  });
  f4.add(text, "distScale", 0.1, 10).onChange(function(val) {
    mdsDistScale = val;
    updateMDSChart();
  });
};
