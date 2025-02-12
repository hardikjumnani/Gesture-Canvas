import cv2
import math
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
import threading
from threading import Thread
import time
from typing import Tuple

pinch: float = 0.0 # 0.0 = open, 0.5 <= pinch

PINCH_RECORDED: bool = False
pinch_dist: float = 0.0

def pixelize_coords(x: float, y: float) -> Tuple[int]:
    h, w, _ = frame.shape
    return (int(x * w), int(y * h))

def distance(coord_1: NormalizedLandmark, coord_2: NormalizedLandmark) -> int:
    x1, y1, z1 = coord_1.x, coord_1.y, coord_1.z
    x2, y2, z2 = coord_2.x, coord_2.y, coord_2.z
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )

def record_pinch() -> None:
    global PINCH_RECORDED, pinch_dist, results
    print("Recording pinch in...")
    for i in range(3, 0, -1):
        print(str(i) + " secs")
        time.sleep(1)
    
    if results:
        hand_landmarks = results.multi_hand_landmarks[0]
        thumb_tip, index_tip = hand_landmarks.landmark[4], hand_landmarks.landmark[8]
        pinch_dist = distance(thumb_tip, index_tip) + 0.01

        print("Pinch Recorded")
        PINCH_RECORDED = True
    else:
        print("Hand not found!")

def check_pinch() -> None:
    global pinch
    dist_index_thumb: int = distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
    if dist_index_thumb < pinch_dist: pinch = 1.0
    else: pinch = max(pinch-0.5, 0.0)

    if pinch == 0.0: print("pinch-open")
    else: print("pinch-pinched")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Set up the hands model
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)
results = None

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Convert BGR to RGB (required by MediaPipe)
    rgb_frame: cv2.typing.MatLike = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Draw landmarks if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            if not PINCH_RECORDED and threading.active_count() <= 1:
                Thread(target=record_pinch).start()
            elif PINCH_RECORDED: check_pinch()

            cv2.line(frame, pixelize_coords(hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y), pixelize_coords(hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y), (0, 255, 0), 6)
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for landmark in [hand_landmarks.landmark[4], hand_landmarks.landmark[8]]:
                x, y, z = landmark.x, landmark.y, landmark.z  # Access coordinates
                
                cv2.circle(frame, pixelize_coords(x, y), 5, (0, 255, 0), 10)
            

    # Display the frame
    cv2.imshow("MediaPipe Hands", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
