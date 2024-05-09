from glob import glob
import concurrent.futures
from utils import *
from producer_config import config as producer_config
from confluent_kafka import Producer
import os
import cv2
import time

class ProducerThread:
    def __init__(self, config):
        self.producer = Producer(config)

    def publishFrame(self, video_path):
        video = cv2.VideoCapture(video_path)
        video_name = video_path.split("/")[-1]
        frame_no = 1
        while video.isOpened():
            _, frame = video.read()
            # pushing every 3rd frame
            if frame_no % 1 == 0:
                frame_bytes = serializeImg(frame)
                self.producer.produce(
                    topic="videostreaming", 
                    value=frame_bytes, 
                    on_delivery=delivery_report,
                    timestamp=frame_no,
                    headers={
                        "video_name": str.encode(video_name)
                    }
                )
                self.producer.poll(0)
            # time.sleep(1)
            frame_no += 1
        video.release()
        return
        
    def start(self, vid_paths):
        # runs until the processes in all the threads are finished
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.publishFrame, vid_paths)

        self.producer.flush() # push all the remaining messages in the queue
        print("Finished...")


if __name__ == "__main__":
    # video_dir = "videos/" ["6449140665","6449005595",""]
    # video_paths = glob(video_dir + "*.mp4") # change extension here accordingly
    rtmp_l =["rtmp://cdn86.cloudvms.in:80/live/6449140665","rtmp://cdn86.cloudvms.in:80/live/6448310845","rtmp://cdn89.cloudvms.in:80/live/6448170098"]
    producer_thread = ProducerThread(producer_config)
    producer_thread.start(rtmp_l)
    