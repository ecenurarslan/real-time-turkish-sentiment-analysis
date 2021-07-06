var me = {};
me.avatar = "https://lh6.googleusercontent.com/-lr2nyjhhjXw/AAAAAAAAAAI/AAAAAAAARmE/MdtfUmC0M4s/photo.jpg?sz=48";

var you = {};
you.avatar = "https://a11.t26.net/taringa/avatares/9/1/2/F/7/8/Demon_King1/48x48_5C5.jpg";

const socket = io();
const classes = {
  positive:{text:'Pozitif', color: 'yellowgreen'},
  notr:{text:'Notr', color: 'lightyellow'},
  negative:{text:'Negatif', color: 'orangered'},
}
var id = 1;
socket.on('prediction', (pred, myId) => {
  if(pred > 0.6){
    pred = classes.positive;
  }else{
    pred = pred < 0.4 ? classes.negative : classes.notr
  }
  var element = document.getElementById(myId);
  var arrowelement = document.getElementById(myId+"-"+myId);
  element.style.backgroundColor = pred.color;
  arrowelement.style.backgroundColor = pred.color;

});
function formatAMPM(date) {
    var hours = date.getHours();
    var minutes = date.getMinutes();
    var ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12;
    hours = hours ? hours : 12; // the hour '0' should be '12'
    minutes = minutes < 10 ? '0'+minutes : minutes;
    var strTime = hours + ':' + minutes + ' ' + ampm;
    return strTime;
}            

//-- No use time. It is a javaScript effect.
function insertChat(who, text, time){
    if (time === undefined){
        time = 0;
    }
    var control = "";
    var date = formatAMPM(new Date());

    if (who == "me"){
      control = '<li style="width:100%;">' +'<div style="float:right;"class="macro">' +
                        '<div style="flex:30; border-top-right-radius:0;" id="'+id+'" class="msj-rta macro">' +
                            '<div class="text text-r">' +
                                '<p>'+text+'</p>' +
                                '<p><small>'+date+'</small></p>' +
                            '</div>' +
                        '<div class="avatar" style="padding:0px 0px 0px 10px !important"><img class="img-circle" style="width:100%;" src="'+you.avatar+'" ></div></div>' +
                        '<div class="aftermessage" id="'+id+'-'+id+'"></div>'+
                    '</div>'+                                
                  '</li>';
                           
    }else{
      control = '<li style="width:100%">' + '<div class="macro"><div class="beforemessage" id="'+id+'-'+id+'"></div>' +
                        '<div style="flex:30; border-top-left-radius:0;"id="'+id+'" class="msj macro">' +
                        '<div class="avatar"><img class="img-circle" style="width:100%;" src="'+ me.avatar +'" /></div>' +
                            '<div class="text text-l">' +
                                '<p>'+ text +'</p>' +
                                '<p><small>'+date+'</small></p>' +
                            '</div>' +
                        '</div> </div>' +
                    '</li>'; 
        
    }
    text = text.toLowerCase();
    text = text.replace(/[^a-zöçğışüİ ]/gi, '')
    socket.emit('chat message', text,id);
    id += 1;
    
    setTimeout(
        function(){                        
            $("ul").append(control).scrollTop($("ul").prop('scrollHeight'));
        }, time);
    
}

function resetChat(){
    $("ul").empty();
}

$(".mytext").on("keydown", function(e){
    if (e.which == 13){
        var text = $(this).val();
        if (text !== ""){
            insertChat("me", text);              
            $(this).val('');
        }
    }
});

$('body > div > div > div:nth-child(2) > span').click(function(){
    $(".mytext").trigger({type: 'keydown', which: 13, keyCode: 13});
})

//-- Clear Chat
resetChat();

//-- Print Messages  
insertChat("you", "Harika bir etkinlikti çok eğlendim", 1500);
insertChat("you", "Sıkıcı ve kötü bir etkinlikti vakit kaybından başka bir şey değil", 8000);


//-- NOTE: No use time on insertChat.