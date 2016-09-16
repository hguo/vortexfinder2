var menuText = function() {
  this.dataName = "Xfieldramp";
  this.frame = 200;
  this.tubeRadius = 0.5;
  this.displayVortexId = true;
  this.displayInclusions = true;
  this.update = function() {};
};

window.onload = function() {
  var text = new menuText();
  var gui = new dat.GUI();
  gui.add(text, 'dataName');
  gui.add(text, 'frame');
  gui.add(text, "tubeRadius", 0.1, 2).onChange(function(val) {
    updateVortexTubes(val);
  })
  gui.add(text, 'displayVortexId').onChange(function(val) {
  	displayVortexId = val;
  })
  gui.add(text, "displayInclusions").onChange(function(on) {
    toggleInclusions(on);
  })
  gui.add(text, 'update');
};
