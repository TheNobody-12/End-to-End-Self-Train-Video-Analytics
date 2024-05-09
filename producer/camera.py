#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import threading
import subprocess

import numpy as np
import cv2


# In[2]:


def add_camera_args(parser):
    """Add parser augument for camera options."""
    parser.add_argument('--rtsp', type=str, default=None,
                        help=('RTSP H.264 stream, e.g. '
                              'rtsp://admin:123456@192.168.1.64:554'))
    return parser


# In[3]:


def open_cam_rtsp(uri):
    """Open an RTSP URI (IP CAM)."""
    return cv2.VideoCapture(uri)


# In[4]:
def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        _, cam.img_handle = cam.cap.read()
        if cam.img_handle is None:
            #logging.warning('Camera: cap.read() returns None...')
            break
    cam.thread_running = False



class Camera():


    def __init__(self, args):
        self.args = args
        self.is_opened = False
        self.thread_running = False
        self.img_handle = None
        self.cap = None
        self.thread = None
        self._open()  # try to open the camera

    def _open(self):
        """Open camera based on command line arguments."""
        if self.cap is not None:
            raise RuntimeError('camera is already opened!')
        a = self.args
        if a.rtsp:
            logging.info('Camera: using RTSP stream %s' % a.rtsp)
            self.cap = open_cam_rtsp(a.rtsp)
            self._start()
        else:
            raise RuntimeError('no camera type specified!')

    def isOpened(self):
        return self.is_opened

    def _start(self):
        if not self.cap.isOpened():
            logging.warning('Camera: starting while cap is not opened!')
            return

        # Try to grab the 1st image and determine width and height
        _, self.img_handle = self.cap.read()
        if self.img_handle is None:
            logging.warning('Camera: cap.read() returns no image!')
            self.is_opened = False
            return

        self.is_opened = True
       
        #self.img_height, self.img_width, _ = self.img_handle.shape
        # start the child thread if not using a video file source
        # i.e. rtsp, usb or onboard
        assert not self.thread_running
        self.thread_running = True
        self.thread = threading.Thread(target=grab_img, args=(self,))
        self.thread.start()

    def _stop(self):
        if self.thread_running:
            self.thread_running = False
            #self.thread.join()

    def read(self):
        """Read a frame from the camera object.

        Returns None if the camera runs out of image or error.
        """
        if not self.is_opened:
            return None

        
        
        return self.img_handle

    def release(self):
        self._stop()
        try:
            self.cap.release()
        except:
            pass
        self.is_opened = False

    def __del__(self):
        self.release()


# In[ ]:




