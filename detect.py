import os
import argparse
from adversarial_detection import AdversarialDetection
import time
import json
from datetime import datetime

import cv2

# Web framework
import socketio
import eventlet
import eventlet.wsgi

import threading

from flask import Flask, send_from_directory
from flask_cors import CORS

# Image Processing
from PIL import Image

# Data IO and Encoding-Decoding
from io import BytesIO
import base64

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from yolov3 import yolov3_anchors, yolov3_tiny_anchors
from yolov3 import yolo_process_output, draw_bounding_box

classes = []
adv_detect = None

# Initialize the camera
camera = cv2.VideoCapture(0)

# Creating the Socket.IO server
sio = socketio.Server(
    cors_allowed_origins='*', 
    async_mode='threading',
    logger=False,  # Disables Socket.IO logging
    engineio_logger=False  # Disables Engine.IO logging
)

# Initialize the flask (web) app
app = Flask(__name__)
app.logger.disabled = True
CORS(app)

# From image to base64 string
def img2base64(image):
    img = Image.fromarray(np.uint8(image))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# Static website
@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
    return send_from_directory(root, path)

@app.route('/', methods=['GET'])
def redirect_to_index():
    return send_from_directory(root, 'index.html')

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    # Immediately send fixed area status to new client
    sio.emit('fixed_area_status', {
        'available': adv_detect.fixed_area is not None
    }, room=sid)

@sio.on('fix_patch')
def fix_patch(sid, data):
    if(data > 0):
        # Stop iterating if we choose to fix the patch
        adv_detect.fixed = True

        # Save each patch
        patch_cv_image = np.zeros((416, 416, 3))
        for box in adv_detect.adv_patch_boxes:
            if adv_detect.monochrome:
                # For black and white images R==G==B
                patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 0] = adv_detect.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 1] = adv_detect.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 2] = adv_detect.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
            else:
                patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :] = adv_detect.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :]

        # Publish the patch image
        sio.emit('patch', {'data': img2base64(patch_cv_image*255.0), 'boxes': adv_detect.adv_patch_boxes})
    else:
        adv_detect.fixed = False

@sio.on('clear_patch')
def clear_patch(sid, data):
    if(data > 0):
        adv_detect.adv_patch_boxes = []
        if adv_detect.monochrome:
            adv_detect.noise = np.zeros((416, 416))
        else:
            adv_detect.noise = np.zeros((416, 416, 3))
        adv_detect.iter = 0
        adv_detect.attack_active = False
        adv_detect.fixed = False

@sio.on('add_patch')
def add_patch(sid, data):
    box = data[1:]
    if(data[0] < 0):
        adv_detect.adv_patch_boxes.append(box)
        adv_detect.iter = 0
    else:
        adv_detect.adv_patch_boxes[data[0]] = box

@sio.on('activate_fixed_area')
def activate_fixed_area(sid, data):
    if adv_detect.fixed_area is not None and data > 0:
        # Activate the fixed area attack
        adv_detect.adv_patch_boxes = [adv_detect.fixed_area]
        adv_detect.iter = 0
        adv_detect.attack_active = True
        adv_detect.fixed = False
        print("Fixed area attack activated")

# Detection thread
def adversarial_detection_thread():  
    global sio, adv_detect, camera, results
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    metrics = None

    while(True): 
        # Capture the video frame
        success, origin_cv_image = camera.read()  # read the camera frame
        if not success:
            print("Failed to open the camera")
            break

        # Read the input image and pulish to the browser
        input_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_BGR2RGB)
        sio.emit('input', {'data': img2base64(input_cv_image)})

        # For YOLO, the input pixel values are normalized to [0, 1]
        input_cv_image = cv2.resize(input_cv_image, (416, 416))
        input_cv_image = input_cv_image.astype(np.float32) / 255.0
    
        start_time = int(time.time() * 1000)
        outs = adv_detect.attack(input_cv_image)

        # Yolo inference
        input_cv_image, outs, stats = adv_detect.attack(input_cv_image)
        boxes, class_ids, confidences = yolo_process_output(outs, yolov3_tiny_anchors, len(classes))

        # Draw bounding boxes
        out_img = draw_bounding_box(input_cv_image, boxes, confidences, class_ids, classes, colors)

        # out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("output", out_img)
        # cv2.waitKey(1)

        elapsed_time = int(time.time()*1000) - start_time
        fps = 1000 / elapsed_time
        print ("fps: ", str(round(fps, 2)))

        if adv_detect.max_iterations is not None:
            # After attack processing, collect metrics
            if adv_detect.attack_active and adv_detect.iter <= adv_detect.max_iterations:
                metrics = adv_detect.collect_attack_metrics()
                results['metrics_over_time'].append(metrics)
            
            # If max iterations reached, save results
            if adv_detect.iter == adv_detect.max_iterations and not adv_detect.fixed:
                results['final_metrics'] = metrics
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = os.path.join(args.output_dir, f"results_{adv_detect.attack_type}_{timestamp}.json")
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2)
                adv_detect.fixed = True
                adv_detect.iter = 0  # Reset iteration count for next experiment
                adv_detect.attack_active = False  # Reset attack status
                print(f"Experiment complete. Results saved to {result_file}")

        # Send the output image to the browser
        sio.emit('adv', {'data': img2base64(out_img*255.0)})

        # Send statistics to the frontend
        sio.emit('stats', {
            'attack_type': stats['attack_type'],
            'original_boxes': stats['original_boxes'],
            'current_boxes': stats['current_boxes'],
            'percentage_increase': stats['percentage_increase'],
            'iterations': stats['iterations']
        })

        eventlet.sleep()

# Websocket thread
def websocket_server_thread():
    global app
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(
    eventlet.listen(('', 9090)),
    app,
    log_output=False,        # Disable access logs
    debug=False             # Disable debug output
    )

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Adversarial Detection')
    parser.add_argument('--model', help='deep learning model', type=str, required=True)
    parser.add_argument('--class_name', help='class names', type=str, required=True)
    parser.add_argument('--attack', help='adversarial attacks type', choices=['one_targeted', 'multi_targeted', 'multi_untargeted'], type=str, required=False, default="multi_untargeted")
    parser.add_argument('--monochrome', action='store_true', help='monochrome patch')
    parser.add_argument('--fixed_area', help='fixed attack area [x,y,w,h]', type=str, default=None)
    parser.add_argument('--max_iter', help='maximum iterations', type=int, default=None)
    parser.add_argument('--output_dir', help='directory to save results', type=str, default='results')
    args = parser.parse_args()

    # Parse fixed area if provided
    fixed_area = None
    if args.fixed_area:
        fixed_area = [int(x) for x in args.fixed_area.split(',')]
        if len(fixed_area) != 4:
            raise ValueError("Fixed area must be in format x,y,w,h")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.class_name) as f:
        content = f.readlines()
    classes = [x.strip() for x in content] 

    # Create a results dictionary
    results = {
        'parameters': vars(args),
        'metrics_over_time': [],
        'final_metrics': None
    }

    try:
        # Initialize with experiment parameters
        adv_detect = AdversarialDetection(
            args.model, 
            args.attack, 
            args.monochrome, 
            classes,
            fixed_area=fixed_area,
            max_iterations=args.max_iter
        )

        t1 = threading.Thread(target=websocket_server_thread, daemon=True)
        t1.start()

        t2 = threading.Thread(target=adversarial_detection_thread, daemon=True)
        t2.start()

        # Main thread will wait here until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Release camera resource
        camera.release()
        # Properly shutdown Socket.IO
        sio.disconnect()
        # Exit the program
        os._exit(0)