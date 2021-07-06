const socket = io();
const predText  = document.getElementById('predicted');
const input = document.getElementById('message');
const classes = {
  positive:{text:'Pozitif', class: 'success'},
  notr:{text:'Notr', class: 'secondary'},
  negative:{text:'Negatif', class: 'danger'},
}
function onSubmit(){
  predText.innerHTML = '';
  comment = input.value;
  input.value = '';
  comment = comment.toLowerCase();
  comment = comment.replace(/[^a-zöçğışüİ ]/gi, '')
  socket.emit('chat message', comment);
}
socket.on('prediction', pred => {
  if(pred > 0.6){
    pred = classes.positive;
  }else{
    pred = pred < 0.4 ? classes.negative : classes.notr
  }
  predText.innerHTML = pred.text
  predText.className = " " + `badge badge-${pred.class}`;
});