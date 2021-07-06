
const pythonScript = 'api.py';
//'dünyanın en iyi filmi harika bayıldım';
const PythonShell = require('python-shell').PythonShell;
const express = require('express')
const app = express()
const port = 3000
const http = require('http');
const server = http.createServer(app);
const path = require('path')
const { Server } = require("socket.io");
const io = new Server(server);

function runPython(input){
  return new Promise((resolve, reject) => {
    let options = {
      mode: 'text',
      pythonOptions: ['-u'], // get print results in real-time
      scriptPath: './',
      args: [input]
    };
    PythonShell.run(pythonScript, options, function (err, results) {
      if (err) throw err;
      resolve(results[0]);
    });
    
  });
}
app.use(express.static('./'))
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});
io.on('connection', (socket) => {
  socket.on('chat message', (msg, id) => {
    console.log('pythona gitti', id,':',msg);
    runPython(msg).then(response => {
      socket.emit('prediction', response, id);
      console.log(response);
    })
  });
});
server.listen(port, ()=>{
  console.log("ready")
})