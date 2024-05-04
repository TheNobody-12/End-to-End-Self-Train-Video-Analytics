from flask import Flask, render_template
from flask_socketio import SocketIO
from pymongo import MongoClient
from utils import insert_data_unique, create_collections_unique
import base64
import requests
import numpy as np
import cv2
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ajith!'
socketio = SocketIO(app, cors_allowed_origins='*')

def display_live_streams():
    client = MongoClient(host='mongodb_connect',
                         port=27017, 
                         username='root', 
                         password='pass',
                         authSource="admin")
    db = client['stream-database']
    video_names = ["stream1", "stream2", "stream3"]
    videos_map = create_collections_unique(db, video_names)

    for rtmp_url in video_names:
        cap = cv2.VideoCapture(rtmp_url)
        if not cap.isOpened():
            print(f"Error: Unable to open stream {rtmp_url}")
            continue
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame (perform object detection, etc.)
            # For demonstration, let's convert the frame to base64 string
            _, img_base64 = cv2.imencode('.jpg', frame)
            img_base64_str = base64.b64encode(img_base64).decode('utf-8')

            # Emit the frame to the client
            socketio.emit('livestream', {'data': [rtmp_url, img_base64_str]})

            # Commit the frame to the database
            frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
            dataframes = {
                "frame": frame_no,
                "videoname": rtmp_url,
                "predictions": []  # You can add object detection predictions here if needed
            }
            videos_map[rtmp_url] = [dataframes]
            insert_data_unique(db, videos_map)

            cv2.waitKey(1)

    cap.release()
    print("Finished...")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def connect():
    display_live_streams()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
