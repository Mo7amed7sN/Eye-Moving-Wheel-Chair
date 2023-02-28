import Models
import io
import socket
import struct
from PIL import Image
import matplotlib.pyplot as pl
import cv2
import numpy as np
from gaze_tracking import GazeTracking
import os

# Socket OBJECT
s = socket.socket()

# Ip and Port for the Connection.
host = '192.168.8.165'
port = 8000

s.bind((host, port))  # ADD IP HERE
s.listen(0)
print('Waiting...')

c, addr = s.accept()

# Load Models...
gaze = GazeTracking()
cam = cv2.VideoCapture(0)

cls = 'Looking center'
tmp = 'Looking center'

# //
mp = dict()
mp['stop'] = 0

while True:

    _, rgb_im = cam.read()

    try:
        gaze.refresh(rgb_im)
        rgb_im = gaze.annotated_frame()
        tmp = Models.Get_Direction1(gaze)
    except:
        pass

    # //
    if gaze.eye_left == None and gaze.eye_right == None:
        mp['stop'] += 1
        if mp['stop'] == 15:
            tmp = 'stop'
    else:
        mp['stop'] = 0

    if tmp is None:
        tmp = cls
    else:
        cls = tmp

    print(tmp)

    try:
        c.send(tmp.encode('utf-8'))
    except:
        break

    cv2.putText(rgb_im, tmp, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    cv2.putText(rgb_im, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(rgb_im, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", rgb_im)

    if cv2.waitKey(1) == 27:
        break

c.close()
s.close()


# tmp_joy2  && final_web_script2