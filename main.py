import cv2
import mediapipe as mp
import pyautogui
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# MediaPipe Setup
# =========================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

# =========================
# Webcam
# =========================

cap = cv2.VideoCapture(0)

cam_w = 1280
cam_h = 720

cap.set(3, cam_w)
cap.set(4, cam_h)

# =========================
# Keyboard Layout
# =========================

keys = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M"],
    ["SPACE","BACK"]
]

final_text = ""

last_click_time = 0
click_delay = 0.5

# =========================
# Draw Keyboard
# =========================

def draw_keyboard(frame):

    button_list = []

    start_x = 50
    start_y = 100

    for i, row in enumerate(keys):

        for j, key in enumerate(row):

            if key == "SPACE":
                key_w = 300
            elif key == "BACK":
                key_w = 180
            else:
                key_w = 80

            key_h = 80

            x = start_x + j * 90
            y = start_y + i * 100

            if key == "BACK":
                x = start_x + 350

            cv2.rectangle(
                frame,
                (x, y),
                (x + key_w, y + key_h),
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                key,
                (x + 15, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            button_list.append((x, y, key, key_w, key_h))

    return button_list

# =========================
# Main Loop
# =========================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    buttons = draw_keyboard(frame)

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            h, w, _ = frame.shape

            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)

            tx = int(thumb_tip.x * w)
            ty = int(thumb_tip.y * h)

            cv2.circle(frame, (ix, iy), 12, (0,255,0), -1)
            cv2.circle(frame, (tx, ty), 12, (0,0,255), -1)

            distance = math.sqrt(
                (index_tip.x - thumb_tip.x) ** 2 +
                (index_tip.y - thumb_tip.y) ** 2
            )

            for button in buttons:

                x, y, key, key_w, key_h = button

                if x < ix < x + key_w and y < iy < y + key_h:

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + key_w, y + key_h),
                        (0,255,0),
                        cv2.FILLED
                    )

                    cv2.putText(
                        frame,
                        key,
                        (x + 15, y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,0),
                        2
                    )

                    current_time = time.time()

                    if distance < 0.03 and current_time - last_click_time > click_delay:

                        if key == "SPACE":
                            pyautogui.press("space")
                            final_text += " "

                        elif key == "BACK":
                            pyautogui.press("backspace")
                            final_text = final_text[:-1]

                        else:
                            pyautogui.press(key.lower())
                            final_text += key

                        last_click_time = current_time

            for landmark in hand_landmarks:

                lx = int(landmark.x * w)
                ly = int(landmark.y * h)

                cv2.circle(frame, (lx, ly), 3, (255,0,0), -1)

    cv2.rectangle(
        frame,
        (50,20),
        (1200,80),
        (50,50,50),
        cv2.FILLED
    )

    cv2.putText(
        frame,
        final_text,
        (60,65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.imshow("AI Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()