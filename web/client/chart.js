function createLineChart() {
  createDummyChart();
  return;
}

function createDummyChart()
{
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
  
  var x = d3.scaleLinear()
      .range([0, width])
      .domain(d3.extent(dataHdrs, function(d) {return d.timestep * dataCfg.dt;}));
  var y = d3.scaleLinear()
      .range([height, 0])
      .domain(d3.extent(dataHdrs, function(d) {return d.V;}));

  var xAxis = d3.axisBottom()
      .scale(x);
  var yAxis = d3.axisLeft()
      .scale(y)
      .ticks(5);
  var line = d3.line()
      .x(function(d) {return x(d.timestep * dataCfg.dt);})
      .y(function(d) {return y(d.V);});
  
  var svg = d3.select("#chart")
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
      .attr("d", line)
}
