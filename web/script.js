// Connect to the WebSocket Server
socket = io.connect('http://' + window.location.hostname + ':9090');

// Receive the input image
socket.on('input', function (data) {
    $('#input').attr("src", "data:image/png;base64," + data.data);
});

// Receive the adv image
socket.on('adv', function (data) {
    $('#adv').attr("src", "data:image/png;base64," + data.data);
});

socket.on('stats', function (data) {
    $('#attackType').text(data.attack_type);
    $('#originalBoxes').text(data.original_boxes);
    $('#currentBoxes').text(data.current_boxes);
    $('#percentageIncrease').text(data.percentage_increase.toFixed(2) + '%');
    $('#iterations').text(data.iterations);
});

// Receive the patch
socket.on('patch', function (data) {
    $('#patch').attr("src", "data:image/png;base64," + data.data);
});

socket.on('connect', function () {
    console.log('Client has connected to the server!');
    clear_patch();
});

socket.on('disconnect', function () {
    console.log('The client has disconnected!');
    $("#customSwitchActivate").prop("checked", false);
});

var boxes = [];

function clear_patch() {
    boxes = [];
    var ctx=$('#canvas')[0].getContext('2d'); 
    ctx.clearRect(0, 0, 416, 416);

    //$("#fixPatchCheck").prop("checked", false);
    socket.emit('fix_patch', 0);

    socket.emit('clear_patch', 1);

    $('#patch').attr("src", "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVQYV2P4DwABAQEAWk1v8QAAAABJRU5ErkJggg==");
}

socket.on('fixed_status', function(data) {
    $("#fixPatchCheck").prop("checked", data.fixed);
});

function fix_patch(fixed) {
    socket.emit('fix_patch', fixed);
}

function activate_fixed_area_attack() {
    // First clear any existing patches
    clear_patch();
    
    // Wait for the clear to complete before activating the fixed area
    socket.once('patch_cleared', function() {
        socket.emit('activate_fixed_area', 1);
    });
};


// Receive the original image
socket.on('update', function (data) {
    $('#origin').attr("src", "data:image/png;base64," + data.data);
});



$(document).ready(function () {
    // Fix patch
    $("#fixPatchCheck").change(function() {
        // Only emit if the change wasn't triggered by the server
        if(this.checked) {
            fix_patch(1);
        }
        else
        {
            fix_patch(0);
        }
    });

    socket.on('fixed_status', function(data) {
        // Mark this as a server update to prevent feedback loop
        $("#fixPatchCheck").data('server-update', true)
                          .prop('checked', data.fixed)
                          .data('server-update', false);
    });

    // Ensure the status handler properly enables the button
    socket.on('fixed_area_status', function(data) {
        const btn = $('#fixedAreaAttackBtn');
        const message = $('#fixedAreaMessage');

        if (data.available) {
            // When fixed area is available
            btn.removeClass('btn-secondary')
            .addClass('btn-primary')
            .prop('disabled', false)
            .css('pointer-events', 'auto')
            message.text('');
        } else {
            btn.removeClass('btn-primary')
            .addClass('btn-secondary')
            .prop('disabled', true)
            .css('pointer-events', 'none');
            message.text('No fixed area specified in command line arguments');
        }
    });

    $(function() {
        var ctx=$('#canvas')[0].getContext('2d'); 
        rect = {};
        drag = false;

        move = false;
        box_index = 0;
        startX = 0;
        startY = 0;

        $(document).on('mousedown','#canvas',function(e){
            startX = e.pageX - $(this).offset().left;
            startY = e.pageY - $(this).offset().top;
            console.log(startX, startY);
            for (box_index = 0; box_index < boxes.length; box_index++) {
                if( (startX >= boxes[box_index].startX) && (startY >= boxes[box_index].startY)) {
                    if( ((startX - boxes[box_index].startX) <= boxes[box_index].w) && ((startY - boxes[box_index].startY) <= boxes[box_index].h)) {
                        move = true;
                        break;
                    }
                }
            }
            if(!move){
                rect.startX = e.pageX - $(this).offset().left;
                rect.startY = e.pageY - $(this).offset().top;
                rect.w=0;
                rect.h=0;
                drag = true;
            }
        });
    
        $(document).on('mouseup', '#canvas', function(){
            // if drawing a rectangle using drag and drop
            if(!move){
                drag = false;
                box = [-1, Math.round(rect.startX), Math.round(rect.startY), Math.round(rect.w), Math.round(rect.h)]
                socket.emit('add_patch', box);
                box = {}
                box.startX = rect.startX
                box.startY = rect.startY
                box.w = rect.w
                box.h = rect.h
                boxes.push(box)
            }
            // If moving an exisiting box, update the position
            else {
                move = false;
                box = [box_index, Math.round(boxes[box_index].startX), Math.round(boxes[box_index].startY), Math.round(boxes[box_index].w), Math.round(boxes[box_index].h)]
                socket.emit('add_patch', box);
            }
        });
    
        $(document).on('mousemove',function(e){
            ctx.clearRect(0, 0, 416, 416);
            boxes.forEach(b => {
                ctx.fillRect(b.startX, b.startY, b.w, b.h);
            });
            // if drawing a rectangle using drag and drop
            if (drag) {
                rect.w = (e.pageX - $("#canvas").offset().left)- rect.startX;
                rect.h = (e.pageY - $("#canvas").offset().top)- rect.startY;
                ctx.fillStyle = 'rgba(0,0,0,0.5)';
                ctx.fillRect(rect.startX, rect.startY, rect.w, rect.h);
            }
            // If moving an exisiting box
            if (move) {
                boxes[box_index].startX += ((e.pageX - $("#canvas").offset().left) - startX);
                boxes[box_index].startY += ((e.pageY - $("#canvas").offset().top) - startY);
                startX = (e.pageX - $("#canvas").offset().left);
                startY = (e.pageY - $("#canvas").offset().top);
            }
        });    
    });
});
