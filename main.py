import cv2
import math
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
import numpy as np
import threading
from threading import Thread
import time
from typing import Tuple, Dict, List, Literal, Any

pinch: float = 0.0 # 0.0 = open, 0.5 <= pinch

PINCH_RECORDED: bool = False
pinch_dist: float = 0.0

CANVAS_OPEN: bool = False
TOOL: Literal["selector", "rectangle", "circle", "line", "heart"] = "" # selector, rectangle, circle, line, heart
DRAWING: bool = False
INIT_PINCH_COORD: Tuple[int] = ()

def pixelize_coords(x: float, y: float) -> Tuple[int]:
    h, w, _ = frame.shape
    return (int(x * w), int(y * h))

def distance(coord_1: NormalizedLandmark, coord_2: NormalizedLandmark) -> int:
    x1, y1, z1 = coord_1.x, coord_1.y, coord_1.z
    x2, y2, z2 = coord_2.x, coord_2.y, coord_2.z
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )

def is_finger_within_box(finger_point: NormalizedLandmark, top_left_coord: Tuple[int], bottom_right_coord: Tuple[int]) -> bool:
    
    finger_x, finger_y = pixelize_coords(finger_point.x, finger_point.y)
    x_between_item: bool = finger_x >= top_left_coord[0] and finger_x <= bottom_right_coord[0]
    y_between_item: bool = finger_y >= top_left_coord[1] and finger_y <= bottom_right_coord[1]
    if x_between_item and y_between_item:
        return True
    else:
        return False

def lerper(min, max): 
    def lerp(norm):
        return (max - min) * norm + min
    return lerp

def draw_heart(frame, top_left_coord: Tuple[int], bottom_right_coord: Tuple[int], color: Tuple[int], thickness: int):
    
    min_x, min_y = top_left_coord
    max_x, max_y = bottom_right_coord
    lerp_x = lerper(min_x, max_x)
    lerp_y = lerper(min_y, max_y)

    # Bezier Curve Points
    p0 = (lerp_x(0.5), lerp_y(1))
    p1 = (lerp_x(0), lerp_y(0.52))
    p2 = (lerp_x(0.125), lerp_y(0))
    p3 = (lerp_x(0.5), lerp_y(0.28))
    p4 = (lerp_x(0.875), lerp_y(0))
    p5 = (lerp_x(1), lerp_y(0.52))

    # Left Curve
    t = 0.0
    while t < 1:
        pX = (1-t)**3 * p0[0] + 3 * (1-t)**2 * t * p1[0] + 3 * (1-t) * t**2 * p2[0] + t**3 * p3[0]
        pY = (1-t)**3 * p0[1] + 3 * (1-t)**2 * t * p1[1] + 3 * (1-t) * t**2 * p2[1] + t**3 * p3[1]
        cv2.circle(frame, (int(pX), int(pY)), 1, color, thickness)

        t += 0.005

    # Right Curve
    t = 0.0
    while t < 1:
        pX = (1-t)**3 * p0[0] + 3 * (1-t)**2 * t * p5[0] + 3 * (1-t) * t**2 * p4[0] + t**3 * p3[0]
        pY = (1-t)**3 * p0[1] + 3 * (1-t)**2 * t * p5[1] + 3 * (1-t) * t**2 * p4[1] + t**3 * p3[1]
        cv2.circle(frame, (int(pX), int(pY)), 1, color, thickness)

        t += 0.005


HUD_ITEMS: Dict[str, List[Tuple[int, int]]] = {
    "three_lines": [((15, 20), (50, 40)), False], # coords, highlighted?
    "selector": [((10, 60), (40, 100)), False],
    "rectangle": [((10, 120), (40, 160)), False],
    "circle": [((10, 180), (50, 220)), False],
    "line": [((10, 240), (40, 280)), False],
    "heart": [((10, 300), (50, 340)), False],
}
def draw_hud(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    if not CANVAS_OPEN:
        cv2.line(frame, (15, 20), (50, 20), (14, 14, 14), 4)
        cv2.line(frame, (15, 30), (50, 30), (14, 14, 14), 4)
        cv2.line(frame, (15, 40), (50, 40), (14, 14, 14), 4)
        return frame

    # selector
    pts = np.array([[10, 60], [10, 100], [20, 90], [38, 93]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

    # rectangle
    cv2.rectangle(frame, (12, 122), (38, 158), (0, 0, 0), 2, cv2.LINE_AA)

    # circle
    cv2.circle(frame, (30, 200), 20, (0, 0, 0), 2, cv2.LINE_AA)

    # line
    cv2.line(frame, (12, 242), (38, 278), (0, 0, 0), thickness=2)

    # heart
    draw_heart(frame, (8, 298), (52, 342), (0, 0, 0), 1)

    return frame

def highlight_hud(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    HIGHLIGHT_MARGIN = 10
    global results
    if not results: return frame
    if not results.multi_hand_landmarks: return frame

    hand_landmarks = results.multi_hand_landmarks[0]
    _, index_tip = hand_landmarks.landmark[4], hand_landmarks.landmark[8]

    for hud_name, item in HUD_ITEMS.items():
        # Three Lines
        if hud_name == "three_lines" and not CANVAS_OPEN:
            hx1, hy1 = item[0][0][0]-HIGHLIGHT_MARGIN, item[0][0][1]-HIGHLIGHT_MARGIN
            hx2, hy2 = item[0][1][0]+HIGHLIGHT_MARGIN, item[0][1][1]+HIGHLIGHT_MARGIN
            
            if is_finger_within_box(index_tip, (hx1, hy1), (hx2, hy2)):
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
                item[1] = True
            else:
                item[1] = False
        
        # All elements
        elif CANVAS_OPEN and not hud_name == "three_lines":
            hx1, hy1 = item[0][0][0]-HIGHLIGHT_MARGIN+5, item[0][0][1]-HIGHLIGHT_MARGIN+5
            hx2, hy2 = item[0][1][0]+HIGHLIGHT_MARGIN-5, item[0][1][1]+HIGHLIGHT_MARGIN-5
            
            if is_finger_within_box(index_tip, (hx1, hy1), (hx2, hy2)):
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
                item[1] = True
            else:
                item[1] = False

    return frame

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

def is_pinched() -> bool:
    global results, pinch
    hand_landmarks = results.multi_hand_landmarks[0]
    thumb_tip, index_tip = hand_landmarks.landmark[4], hand_landmarks.landmark[8]
    dist_index_thumb: int = distance(thumb_tip, index_tip)

    if dist_index_thumb < pinch_dist: pinch = 1.0
    else: pinch = max(pinch-0.5, 0.0)

    if pinch == 0.0: return False
    else: return True

def action_on_pinch(frame: cv2.typing.MatLike) -> None:
    global CANVAS_OPEN, DRAWING, TOOL, INIT_PINCH_COORD, results

    if not results or not results.multi_hand_landmarks: return None
    if not is_pinched(): return None

    index_tip = results.multi_hand_landmarks[0].landmark[8]
    curr_finger_coord: Tuple[int] = pixelize_coords(index_tip.x, index_tip.y)

    # Three Lines
    if not CANVAS_OPEN:
        if DRAWING:
            if TOOL == "selector": ...
            elif TOOL == "rectangle": 
                if not INIT_PINCH_COORD:
                    INIT_PINCH_COORD = curr_finger_coord
                else:
                    cv2.rectangle(frame, INIT_PINCH_COORD, curr_finger_coord, (200, 0, 0), 2)

            elif TOOL == "circle": 
                if not INIT_PINCH_COORD:
                    INIT_PINCH_COORD = curr_finger_coord
                else:
                    ix, iy = INIT_PINCH_COORD
                    cx, cy = curr_finger_coord
                    cv2.circle(frame, ((cx+ix)//2, (cy+iy)//2), abs(cx-ix)//2, (200, 0, 0), 2)

            elif TOOL == "line": 
                if not INIT_PINCH_COORD:
                    INIT_PINCH_COORD = curr_finger_coord
                else:
                    cv2.line(frame, INIT_PINCH_COORD, curr_finger_coord, (200, 0, 0), 2)
            
            elif TOOL == "heart":
                if not INIT_PINCH_COORD:
                    INIT_PINCH_COORD = curr_finger_coord
                else:
                    draw_heart(frame, INIT_PINCH_COORD, curr_finger_coord, (228, 12, 204), 1)
        
        if is_finger_within_box(index_tip, HUD_ITEMS["three_lines"][0][0], HUD_ITEMS["three_lines"][0][1]):
            CANVAS_OPEN = True
            DRAWING = False
            TOOL = ""
            INIT_PINCH_COORD = ()

    elif CANVAS_OPEN and not DRAWING:
        for hud_name, item in HUD_ITEMS.items():
            if hud_name == "three_lines": continue
            if is_finger_within_box(index_tip, item[0][0], item[0][1]):
                DRAWING = True
                TOOL = hud_name
                CANVAS_OPEN = False

CANVAS: List[Dict[str, Any]] = []
def action_off_pinch(frame: cv2.typing.MatLike) -> None:
    global CANVAS_OPEN, DRAWING, TOOL, INIT_PINCH_COORD, results, CANVAS

    if not results or not results.multi_hand_landmarks: return None
    if is_pinched(): return None

    index_tip = results.multi_hand_landmarks[0].landmark[8]

    if not CANVAS_OPEN:
        if DRAWING:
            if INIT_PINCH_COORD:
                # save canvas object
                object = {
                    "type": TOOL,
                    "top_left_coord": INIT_PINCH_COORD,
                    "bottom_right_coord": pixelize_coords(index_tip.x, index_tip.y)
                }
                CANVAS.append(object)

                INIT_PINCH_COORD = ()

def draw_canvas_objects(frame: cv2.typing.MatLike) -> None:
    global CANVAS
    for object in CANVAS:
        if object["type"] == "rectangle":
            cv2.rectangle(frame, object["top_left_coord"], object["bottom_right_coord"], (200, 0, 0), 2)
        elif object["type"] == "circle":
            ix, iy = object["top_left_coord"]
            cx, cy = object["bottom_right_coord"]
            cv2.circle(frame, ((cx+ix)//2, (cy+iy)//2), abs(cx-ix)//2, (200, 0, 0), 2)
        elif object["type"] == "line":
            cv2.line(frame, object["top_left_coord"], object["bottom_right_coord"], (200, 0, 0), 2)
        elif object["type"] == "heart":
            draw_heart(frame, object["top_left_coord"], object["bottom_right_coord"], (228, 12, 204), 1)
    
    

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
            elif PINCH_RECORDED: is_pinched()

            if TOOL == "": cv2.line(frame, pixelize_coords(hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y), pixelize_coords(hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y), (0, 255, 0), 6)
            else: cv2.line(frame, pixelize_coords(hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y), pixelize_coords(hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y), (255, 0, 0), 2)
            # Draw hand landmarks on the frame
            # mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for landmark in [hand_landmarks.landmark[4], hand_landmarks.landmark[8]]:
                x, y, z = landmark.x, landmark.y, landmark.z  # Access coordinates
                
                cv2.circle(frame, pixelize_coords(x, y), 5, (0, 255, 0), 10)
            

    # Display the frame
    frame = draw_hud(frame)
    frame = highlight_hud(frame)
    action_on_pinch(frame)
    action_off_pinch(frame)

    draw_canvas_objects(frame)
    cv2.imshow("MediaPipe Hands", frame)

    key = cv2.waitKey(1) & 0xFF
    # Press 'q' to exit
    if key == ord('q'):
        break
    # Press 'z' to undo a shape
    elif key == ord('z'):
        CANVAS.pop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
