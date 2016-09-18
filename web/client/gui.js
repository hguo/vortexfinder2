var menuText = function() {
  this.dataName = "GL_3D_Xfieldramp_inter";
  this.frame = 200;
  this.resetTrackball = function () {
    cameraControls.reset();
  };
  this.displayStats = false;
  this.displayVoltage = true;
  this.displayVortexId = false;
  this.displayInclusions = true;
  this.displayMDS = false;
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
};

window.onload = function() {
  var text = new menuText();
  var gui = new dat.GUI();

  var f1 = gui.addFolder("Data");
  f1.add(text, 'dataName');
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
  f2.add(text, "resetTrackball");
  f2.open();

  var f3 = gui.addFolder("Charts");
  f3.add(text, "displayVoltage").onChange(function(val) {
    if (val) $("#voltageChart").css({visibility: "visible"});
    else $("#voltageChart").css({visibility: "hidden"});
  });
  f3.add(text, "displayMDS").onChange(function(val) {
    displayMDS = val;
    if (val) {
      updateMDSChart();
      $("#mdsChart").css({visibility: "visible"});
    } else {
      $("#mdsChart").css({visibility: "hidden"});
    }
  })
  f3.add(text, "distScale", 0.1, 10).onChange(function(val) {
    mdsDistScale = val;
    updateMDSChart();
  });
};
