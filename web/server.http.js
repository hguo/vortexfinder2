var http = require("http");
var url = require("url");
var querystring = require("querystring");
// var vf2 = require('bindings')('vf2');

const dbname="GL_3D_Xfieldramp_inter.rocksdb";

http.createServer(function(request, response) {
  var parse = url.parse(request.url);
  var pathname = parse.pathname;
  console.log("Request for " + pathname + " received.");
  
  response.writeHead(200, {"Content-Type": "text/plain"});
  response.write(pathname);
  response.end();
   
  var queries = querystring.parse(parse.query);

  if (pathname == "/requestDataInfo") {
    queryDataInfo();
  } else if (pathname == "/requestFrame") {
    queryFrame(queries.frame);
  }
}).listen(8080);

function queryDataInfo() {
}

function queryFrame(frame) {
  console.log("load frame " + frame);
}
