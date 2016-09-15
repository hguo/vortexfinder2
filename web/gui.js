var menuText = function() {
  this.dataName = "Xfieldramp";
  this.frame = 200;
  this.displayVortexId = true;
  this.update = function() {};
};

window.onload = function() {
  var text = new menuText();
  var gui = new dat.GUI();
  gui.add(text, 'dataName');
  gui.add(text, 'frame');
  gui.add(text, 'displayVortexId').onChange(function(val) {
  	displayVortexId = val;
  })
  gui.add(text, 'update');
};