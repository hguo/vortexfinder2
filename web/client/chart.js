function createLineChart() {
  var W = window.innerWidth, H = 120;
  var margin = {top: 20, right: 20, bottom: 30, left: 50},
      width = W - margin.left - margin.right,
      height = H - margin.top - margin.bottom;
  
  $("svg").css({
    top: window.innerHeight - 120, 
    left: 0, 
    width: window.innerWidth, 
    height: 120, 
    position: "absolute"
  });
 
  const dt = dataCfg.dt;
  var x = d3.scaleLinear()
      .range([0, width])
      .domain(d3.extent(dataHdrs, function(d) {return d.timestep * dt;}));
  var y = d3.scaleLinear()
      .range([height, 0])
      .domain(d3.extent(dataHdrs, function(d) {return d.V;}));

  var xAxis = d3.axisBottom()
      .scale(x);
  var yAxis = d3.axisLeft()
      .scale(y)
      .ticks(5);
  var line = d3.line()
      .x(function(d) {return x(d.timestep * dt);})
      .y(function(d) {return y(d.V);});
  
  var svg = d3.select("#voltageChart")
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
  svg.append("g")
      .attr("class", "xaxis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
      .append("text")
      .attr("x", width)
      .attr("y", 15)
      .text("Time")
  svg.append("g")
      .attr("class", "yaxis")
      .call(yAxis)
      .append("text")
      .attr("x", 60)
      .attr("dy", ".71em")
      .style("text-anchor", "begin")
      .text("Voltage");
  svg.append("path")
      .datum(dataHdrs)
      .attr("class", "line")
      .attr("d", line);

  var cursor = svg.append("g")
      .attr("class", "cursor")
      .style("display", "none"); // TODO

  var focus = svg.append("g")
      .attr("class", "focus")
      .style("display", "none");
  focus.append("circle")
      .attr("class", "circle")
      .attr("r", 4.5);
  focus.append("text")
      .attr("y", -15)
      .attr("dy", ".35em");

  svg.append("rect")
      .attr("class", "overlay")
      .attr("width", width)
      .attr("height", height)
      .on("mouseover", function() {focus.style("display", null);})
      .on("mouseout", function() {focus.style("display", "none");})
      .on("mousemove", mousemove);

  function mousemove(val) {
    var x0 = x.invert(d3.mouse(this)[0]);
    var bisect = d3.bisector(function(d) {return d.timestep*dt;}).left;
    var i = bisect(dataHdrs, x0);
    var hdr = dataHdrs[i];

    // var x0 = d3.mouse(this)[0];
    // i = d3.bisector(function(d) {return d.timestep * dt;}).left,
    //     d = dataHdrs[i];
    // focus.attr("transform", "translate(" + x(d.timestep) + "," + y(d.V) + ")");
    focus.attr("transform", "translate(" + x(x0) + "," + y(hdr.V) + ")");
    focus.select("text").text(hdr.V.toFixed(3));
  }
}
