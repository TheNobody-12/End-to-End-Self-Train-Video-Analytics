import cv2
from utils import serializeImg
from producer_config import config as producer_config
from confluent_kafka import Producer
import time

class RTMPProducer:
    def __init__(self, config):
        self.producer = Producer(config)

    def publishFrames(self, rtmp_url):
        cap = cv2.VideoCapture(rtmp_url)
        frame_no = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # pushing every 3rd frame
            if frame_no % 3 == 0:
                frame_bytes = serializeImg(frame)
                self.producer.produce(
                    topic="videostreaming",
                    value=frame_bytes,
                    on_delivery=self.delivery_report,
                    timestamp=frame_no,
                    headers={
                        "rtmp_url": str.encode(rtmp_url)
                    }
                )
            frame_no += 1
            time.sleep(1 / 24)  # Adjust the frame rate to 24 frames per second (or your desired rate)
        cap.release()

    def delivery_report(self, err, msg):
        if err is not None:
            print(f"Message delivery failed: {err}")
        else:
            print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def start(self, rtmp_urls):
        for rtmp_url in rtmp_urls:
            self.publishFrames(rtmp_url)

        self.producer.flush()
        print("Finished...")


if __name__ == "__main__":
    rtmp_streams = ["rtmp://cdn22.cloudvms.in:80/dvr1/anpr", "rtmp://cdn22.cloudvms.in:80/dvr1/anpr", "rtmp://cdn22.cloudvms.in:80/dvr1/anpr"]

    producer = RTMPProducer(producer_config)
    producer.start(rtmp_streams)
