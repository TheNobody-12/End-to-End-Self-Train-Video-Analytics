import cv2
from ultralytics import YOLO
import argparse
import time
import numpy as np
from camera import add_camera_args, Camera
from collections import defaultdict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing ')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument("-y", "--yolo", required=True,
       help="base path to YOLO directory")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
       help="minimum probability to filter weak detections")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
       help="threshold when applyong non-maxima suppression")
    parser.add_argument("-C", "--CamDID", required=True,
        help="Camera DID for AI")
    parser.add_argument("-Objapi", "--HeadApi", required=True, 
        help="Object / Head API for camera")
    parser.add_argument("-url","--url",required=True,help="url alert to be sent")
    parser.add_argument("-time_int","-time_int",type=int,default = 30, help="Time interval for sending request to server")
    args = parser.parse_args()
    return args

args = parse_args()
cam = Camera(args)

classes = [0]

print(args.yolo)

start = 0
end = 0

camid = args.CamDID
url = args.HeadApi
time_diff = args.time_int


class date:
	prev_date=''
class wait_count:
	current_time=0
	start_time=0
	current_time1=0
	start_time1=0
	prev_object_count=0
	prev_person_count=0

# Loading the YOLOv8 model
print("[INFO] loading YOLOv8 model...")
model_path = "yolov8m.pt"
model = YOLO(model_path)
usleep = lambda x: time.sleep(x/1000000.0)


# Store the track history
track_history = defaultdict(lambda: [])

while True:
    person_count = 0

    frame = cam.read()

    if frame is None:
        print("Image cannot be read")
        cam.release()
        cam = Camera(args)
        print("Stream ended. Displaying message.")
        cv2.putText(frame, "Stream ended", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)  # Wait for a key press before breaking out of the loop
        break
    
    frame = cv2.resize(frame, (640, 640),interpolation = cv2.INTER_AREA)
    # cv2.imwrite("frame.jpg",frame)
    start = time.time()
    # Run inference on an image
    results = model.track(frame,persist=True,device=0, classes=classes,conf=0.2,imgsz=[640,640])
    for r in results:
        boxes = r.boxes.xywh.tolist()
        track_ids = r.boxes.id
        if track_ids is not None:
            track_ids = track_ids.int().tolist()
        else:
            track_ids = []
        annotated_frame = r.plot()

        # Plot the tracks and also check if it is comming towards the camera or going away
        if len(track_ids) > 0 or track_ids is not None:
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                # Check if the track is going away or towards the camera and draw tracking line accordingly
                if len(track) > 1:
                    x_diff = track[-1][0] - track[0][0]
                    y_diff = track[-1][1] - track[0][1]
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                    # if x_diff > 0:
                    #     cv2.putText(annotated_frame, "Going away", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    #     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)
                    # else:
                    #     cv2.putText(annotated_frame, "Coming towards", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    #     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)

     
    # time.sleep(1) #sleep during 100μs

    end = time.time()
    sec = end - start
    fps = 1 / (sec)
    fps = str(fps)
    print("FPS: ", fps)

    # cv2.putText(frame, "Person count: " + str(person_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.putText(frame, "FPS: " + fps, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Display the annotated frame
    # cv2.imshow("YOLOv8 Tracking", annotated_frame)


    # for r in results:
    #     boxes = r.boxes.xyxy.tolist()
    #     t_boxes = r.boxes.xywh.cpu()
    #     labels = r.boxes.cls.tolist()
    #     track_ids = r.boxes.id.int().cpu().tolist()
    #     conf = r.boxes.conf.tolist()
    #     annotated_frame = r.plot()


    # for i in range(len(labels)):
    #     if labels[i] == 0:
    #         person_count += 1
    #         cv2.rectangle(frame, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)
    #         # cv2.putText(frame, str(conf[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         # cv2.putText(frame, str(labels[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     else:
    #         object_count=object_count+1

    # print("Person count: ", person_count)

   

    # cv2.imshow("Frame", frame)

    
     # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
