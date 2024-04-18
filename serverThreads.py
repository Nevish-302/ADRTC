import pickle
import socket
import struct
import time
import threading
# Loading the Image

# img=cv2.imread(r'E:\pycharm\venv\hackcbs1.jpg',0)

# img=cv2.resize(img,(400,400))
# img=cv2.resize(img,(0,0),fx=0.5,fy=0.5)
# print(type(img))
#
# for i in range(10):server.py
#     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

import cv2

HOST = ''
PORT = 8089

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

model=load_model("./newmod.h5")

conn, addr = s.accept()

data = b'' ### CHANGED
payload_size = struct.calcsize("L") ### CHANGED


#
import time

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

model=load_model("./nmod.h5")

mp_body=mp.solutions.holistic
mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
mp_drawstyle=mp.solutions.drawing_styles

body=mp_body.Holistic(min_detection_confidence=0.8,min_tracking_confidence=0.8)
hands=mp_hands.Hands(max_num_hands=10,min_detection_confidence=0.8,min_tracking_confidence=0.8)

cap=cv2.VideoCapture(0)
drawspecs=mp_draw.DrawingSpec(thickness=1,circle_radius=2,color=[255, 0,0])


actions=['again','boy','deaf','girl','hard of hearing','hearing','help','how','know','me','meet']
no_sequence=20

videos=20

#
frames = []
seq = []

i = 0

answer = ''
def analyze():
    global frames
    global seq
    global answer
    while True:
        if len(seq) < 20:
            for frame in frames:
                framer = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_pose = body.process(framer)
                pose = np.zeros((33, 4))
                if result_pose.pose_landmarks:
                    mp_draw.draw_landmarks(framer, result_pose.pose_landmarks, mp_body.POSE_CONNECTIONS, drawspecs, drawspecs)
                    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in result_pose.pose_landmarks.landmark])

                left = np.zeros((21, 4))
                if result_pose.left_hand_landmarks:
                    mp_draw.draw_landmarks(framer, result_pose.left_hand_landmarks, mp_body.HAND_CONNECTIONS, drawspecs,
                                           drawspecs)
                    left = np.array([[l.x, l.y, l.z, l.visibility] for l in result_pose.left_hand_landmarks.landmark])

                right = np.zeros((21, 4))
                if result_pose.right_hand_landmarks:
                    mp_draw.draw_landmarks(framer, result_pose.right_hand_landmarks, mp_body.HAND_CONNECTIONS, drawspecs,
                                           drawspecs)
                    right = np.array([[l.x, l.y, l.z, l.visibility] for l in result_pose.right_hand_landmarks.landmark])

                a = np.concatenate([left, right, pose])
                seq.append(a)
            frames = []

            answer = '???'
            # cv2.imshow('frame', frame)
        else:
            input_data = np.expand_dims(np.array(seq[-20:], dtype=np.float32), axis=0)
            seq = []
            input_data = input_data.reshape(1, 20, 300)
            v = model.predict(input_data).squeeze()

            # cv2.putText(framer, f'{actions[np.argmax(v)]}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=1,
            #             color=(255, 255, 255), thickness=1)
            answer = actions[np.argmax(v)]
            print(actions[np.argmax(v)])


def show():
    global answer
    global frames
    global seq
    global data
    global i
    while True:

        # Retrieve message size
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

        # Retrieve all data based on message size
        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]
        i += 1
        # Extract frame
        starttime=time.time()
        framer = pickle.loads(frame_data)
        v=0
        frames.append(framer)

        cv2.putText(framer, f'{answer}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=1,
                                 color=(255, 255, 255), thickness=1)
        cv2.imshow('frame', framer)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 800, 600)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
thread_one = threading.Thread(target=analyze)
thread_two = threading.Thread(target=show)
thread_two.start()
thread_one.start()
thread_one.join()
thread_two.join()
#from tqdm import tqdm
#import requests

#url = "http://download.thinkbroadband.com/10MB.zip"
#response = requests.get(url, stream=True)

#with open("10MB", "wb") as handle:
 #   for data in tqdm(response.iter_content()):
  #      handle.write(data)
