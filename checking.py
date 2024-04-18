
import time
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model=load_model("./newmod.h5")

mp_body=mp.solutions.holistic
mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
mp_drawstyle=mp.solutions.drawing_styles

body=mp_body.Holistic(min_detection_confidence=0.8,min_tracking_confidence=0.8)
hands=mp_hands.Hands(max_num_hands=10,min_detection_confidence=0.8,min_tracking_confidence=0.8)

cap=cv2.VideoCapture(0)
drawspecs=mp_draw.DrawingSpec(thickness=1,circle_radius=2,color=[255, 0,0])

# 'biscuits','current'
actions=[

    'again','boy','deaf',
'girl','hard of hearing','hearing','help','how','know','me','meet'

    ]

no_sequence=20

videos=20

v=0

seq=[]
while cap.isOpened():

    ret, frame = cap.read()
    framer = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_pose = body.process(framer)
    framer = cv2.cvtColor(framer, cv2.COLOR_RGB2BGR)

    pose = np.zeros((33, 4))
    if result_pose.pose_landmarks:
        mp_draw.draw_landmarks(framer, result_pose.pose_landmarks, mp_body.POSE_CONNECTIONS, drawspecs, drawspecs)
        pose = np.array([[l.x, l.y, l.z, l.visibility] for l in result_pose.pose_landmarks.landmark])

    left = np.zeros((21, 4))
    if result_pose.left_hand_landmarks:
        mp_draw.draw_landmarks(framer, result_pose.left_hand_landmarks, mp_body.HAND_CONNECTIONS, drawspecs, drawspecs)
        left = np.array([[l.x, l.y, l.z, l.visibility] for l in result_pose.left_hand_landmarks.landmark])
    # print(left)

    right = np.zeros((21, 4))
    if result_pose.right_hand_landmarks:
        mp_draw.draw_landmarks(framer, result_pose.right_hand_landmarks, mp_body.HAND_CONNECTIONS, drawspecs, drawspecs)
        right = np.array([[l.x, l.y, l.z, l.visibility] for l in result_pose.right_hand_landmarks.landmark])

    a = np.concatenate([left, right, pose])
    seq.append(a)



    if len(seq)<20:
        continue

    input_data = np.expand_dims(np.array(seq[-20:], dtype=np.float32), axis=0)
    input_data = input_data.reshape(1, 20, 300)
    # print(input_data.shape)
    v = model.predict(input_data).squeeze()
        # print(v)
    print(np.argmax(v))

    cv2.putText(framer, f'{actions[np.argmax(v)]}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255), thickness=1)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 800, 600)
    cv2.imshow('frame', framer)


    if cv2.waitKey(2) == ord('r'):
        break



#------------------------------------------------------- Model Prediction Part ------------------------------------




cap.release()
cv2.destroyAllWindows()

