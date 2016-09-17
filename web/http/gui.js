var menuText = function() {
  this.dataName = "GL_3D_Xfieldramp_inter";
  this.frame = 200;
  this.tubeRadius = 0.5;
  this.displayVortexId = false;
  this.displayInclusions = true;
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
  gui.add(text, 'dataName');
  gui.add(text, 'frame').onChange(function(val) {
    currentFrame = val;
    requestCurrentFrame();
  });
  gui.add(text, "tubeRadius", 0.1, 2).onChange(function(val) {
    updateVortexTubes(val);
  })
  gui.add(text, 'displayVortexId').onChange(function(val) {
    toggleVortexId(val);
  })
  gui.add(text, "displayInclusions").onChange(function(on) {
    toggleInclusions(on);
  })
  gui.add(text, 'previousFrame');
  gui.add(text, 'nextFrame');
};
