var xScale = {};

function createLineChart() {
  var W = window.innerWidth, H = 120;
  var margin = {top: 20, right: 20, bottom: 30, left: 50},
      width = W - margin.left - margin.right,
      height = H - margin.top - margin.bottom;
  
  $("#voltageChart").css({
    top: window.innerHeight - 120, 
    left: 0, 
    width: window.innerWidth, 
    height: 120, 
    position: "absolute"
  });
 
  const dt = dataCfg.dt;
  xScale = d3.scaleLinear()
      .range([0, width])
      .domain(d3.extent(dataHdrs, function(d) {return d.timestep * dt;}));
  var yScale = d3.scaleLinear()
      .range([height, 0])
      .domain(d3.extent(dataHdrs, function(d) {return d.V;}));

  var xAxis = d3.axisBottom()
      .scale(xScale);
  var yAxis = d3.axisLeft()
      .scale(yScale)
      .ticks(3);
  var line = d3.line()
      .x(function(d) {return xScale(d.timestep * dt);})
      .y(function(d) {return yScale(d.V);});
  
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

  var frameCursor = svg.append("g")
      .attr("id", "frameCursor")
      .attr("transform", "translate(" + xScale(dataHdrs[currentFrame].timestep*dt) + ",0)")
      .append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", height)
      .style("stroke", "steelblue");

  var focus = svg.append("g")
      .attr("class", "focus")
      .style("display", "none");
  focus.append("circle")
      .attr("class", "circle")
      .attr("r", 4.5);
  focus.append("text")
      .attr("y", -15)
      .attr("dy", ".35em");
  
  var cursor = svg.append("g");
  cursor.append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", height)
      .style("stroke", "steelblue")
      .style("stroke-dasharray", "2,2");

  svg.append("rect")
      .attr("class", "overlay")
      .attr("width", width)
      .attr("height", height)
      .on("mouseover", function() {
        focus.style("display", null);
        cursor.style("display", null);
      })
      .on("mouseout", function() {
        focus.style("display", null);
        cursor.style("display", null);
      })
      .on("mousemove", mousemove)
      .on("click", click);

  function mousemove(val) {
    var x0 = xScale.invert(d3.mouse(this)[0]);
    var bisect = d3.bisector(function(d) {return d.timestep*dt;}).left;
    var i = bisect(dataHdrs, x0);
    var hdr = dataHdrs[i];

    // var x0 = d3.mouse(this)[0];
    // i = d3.bisector(function(d) {return d.timestep * dt;}).left,
    //     d = dataHdrs[i];
    // focus.attr("transform", "translate(" + x(d.timestep) + "," + y(d.V) + ")");
    focus.attr("transform", "translate(" + xScale(x0) + "," + yScale(hdr.V) + ")");
    focus.select("text").text(hdr.V.toFixed(3));
    cursor.attr("transform", "translate(" + xScale(x0) + ",0)");
  }

  function click(val) {
    var x0 = xScale.invert(d3.mouse(this)[0]);
    var bisect = d3.bisector(function(d) {return d.timestep*dt;}).left;
    var i = bisect(dataHdrs, x0);
    currentFrame = i;
    requestFrame(i);
  }
}

function createMDSChart() {
}

function updateMDSChart() {
  const W = 320, H = 320;
  var svg = d3.select("#mdsChart");

  var nodes = d3.range(vortexId.length).map(function(i) {
    return {
      id: vortexId[i],
      color: vortexColorsHex[i]
    };
  });

  const distScale = 2;

  var forceLink = d3.forceLink(vortexDistances)
    .id(function(d) {return d.id;})
    .distance(function(d) {return d.dist * distScale;});

  var simulation = d3.forceSimulation(nodes)
    .force("charge", d3.forceManyBody())
    .force("link", forceLink)
    .force("center", d3.forceCenter(W/2, H/2))
    .on("tick", ticked);

  svg.select(".nodes").remove();
  var node = svg.append("g")
    .attr("class", "nodes")
    .selectAll("circle")
    .data(nodes)
    .enter().append("circle")
    .attr("r", 3)
    .attr("fill", function(d) {return d.color;});

  function ticked() {
    nodes.forEach(function(d) {
      node.attr("cx", function(d) {return d.x;});
      node.attr("cy", function(d) {return d.y;});
    });
  }
}
