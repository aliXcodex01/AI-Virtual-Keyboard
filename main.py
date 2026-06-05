"""
AI Virtual Keyboard  —  Hand-Gesture Controlled
================================================
Pinch index finger + thumb together over a key to type.

SETUP
-----
1.  pip install opencv-python mediapipe pyautogui

2.  Download model (once, place in same folder as this file):
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/
    hand_landmarker/float16/latest/hand_landmarker.task

3.  python virtual_keyboard.py

4.  Press Q to quit.

BUGS FIXED IN THIS VERSION
---------------------------
BUG 1 – Hand:NO / FPS 0.9 (blocking detection)
    detect_for_video() requires STRICTLY INCREASING timestamps.
    time.time() on Windows has ~15 ms clock resolution, so consecutive
    frames in the same tick all got timestamp=0 → MediaPipe blocked.
    Fix: use  frame_count * 33  (always 0, 33, 66 … no duplicates).

BUG 2 – Black camera feed
    DSHOW on Windows sometimes delivers raw YUV/YUYV frames.
    Fix: set CAP_PROP_CONVERT_RGB=1 immediately after opening.

BUG 3 – NORM_RECT landmark warning → broken tracking
    detect() (IMAGE mode) assumes a square ROI on non-square frames.
    Fix: RunningMode.VIDEO + detect_for_video() handles any aspect ratio.
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import math
import sys
import os

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

CAMERA_INDEX   = 0       # change to 1 or 2 if wrong camera opens
CAM_W          = 1280
CAM_H          = 720
MODEL_PATH     = "hand_landmarker.task"

PINCH_THRESH   = 0.040   # lower = tighter pinch needed to click
CLICK_COOLDOWN = 0.50    # seconds between key presses
FLASH_SECS     = 0.18

KEY_W          = 78
KEY_H          = 72
KEY_GAP        = 8
KB_Y           = 108     # keyboard top edge

# BGR colours
C_KEY_BG     = ( 25,  25,  25)
C_KEY_BOR    = (110, 110, 110)
C_HOVER      = ( 40, 110,  40)
C_FLASH      = ( 30, 210,  80)
C_SHIFT_ON   = ( 90,  60, 160)
C_WHITE      = (255, 255, 255)
C_BLACK      = (  0,   0,   0)
C_INDEX      = (  0, 230,   0)
C_THUMB      = (  0,   0, 230)
C_BONE       = (180, 140,  90)
C_BAR_BG     = ( 18,  18,  18)
C_BAR_TXT    = (140, 215, 140)

# ═══════════════════════════════════════════════════════════════════
# KEYBOARD LAYOUT
# ═══════════════════════════════════════════════════════════════════

ROWS = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["SHIFT","Z","X","C","V","B","N","M","BACK"],
    ["SPACE"],
]
WIDE = {"SHIFT": 1.7, "BACK": 1.7, "SPACE": 8.0}


def build_buttons(cam_w):
    buttons = []
    for ri, row in enumerate(ROWS):
        widths    = [int(WIDE.get(k, 1.0) * KEY_W) for k in row]
        row_w     = sum(widths) + KEY_GAP * (len(row) - 1)
        x0        = (cam_w - row_w) // 2
        y         = KB_Y + ri * (KEY_H + KEY_GAP)
        cx        = x0
        for key, w in zip(row, widths):
            buttons.append(dict(key=key, x=cx, y=y, w=w, h=KEY_H))
            cx += w + KEY_GAP
    return buttons


# ═══════════════════════════════════════════════════════════════════
# SHIFT STATE
# ═══════════════════════════════════════════════════════════════════

class ShiftState:
    OFF, ONCE, CAPS = 0, 1, 2

    def __init__(self):
        self.s  = self.OFF
        self._t = 0.0

    def toggle(self):
        now = time.time()
        if now - self._t < 0.45:
            self.s = self.CAPS if self.s != self.CAPS else self.OFF
        else:
            self.s = self.ONCE if self.s == self.OFF else self.OFF
        self._t = now

    def consume(self):
        if self.s == self.ONCE:
            self.s = self.OFF

    @property
    def active(self): return self.s != self.OFF

    @property
    def label(self): return {0:"off", 1:"SFT", 2:"CAP"}[self.s]


# ═══════════════════════════════════════════════════════════════════
# CAMERA  (Windows DSHOW, force RGB conversion)
# ═══════════════════════════════════════════════════════════════════

def open_camera(idx, want_w, want_h):
    is_win   = sys.platform.startswith("win")
    backends = ([("DSHOW", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)]
                if is_win else [("default", cv2.CAP_ANY)])

    for bname, bflag in backends:
        for rw, rh in [(want_w, want_h), (1280, 720), (640, 480)]:
            print(f"[cam] {bname}  {rw}x{rh} ...")
            cap = cv2.VideoCapture(idx, bflag)
            if not cap.isOpened():
                cap.release(); continue

            # Force RGB conversion – DSHOW sometimes gives raw YUV (looks black)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  rw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)

            time.sleep(0.5)
            for _ in range(20):
                ok, frm = cap.read()
                if ok and frm is not None and frm.size > 0:
                    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[cam] opened  {bname}  actual={aw}x{ah}")
                    return cap
                time.sleep(0.05)
            cap.release()

    sys.exit(
        "[ERROR] Cannot open camera.\n"
        "  • Close Teams / Zoom / OBS / Windows Camera app.\n"
        "  • Try CAMERA_INDEX = 1 or 2 at the top of this file.\n"
        "  • Check Device Manager > Imaging devices for driver errors.\n"
        "  • Windows Settings > Privacy > Camera > allow Desktop apps."
    )


# ═══════════════════════════════════════════════════════════════════
# MEDIAPIPE  (VIDEO mode — handles non-square frames correctly)
# ═══════════════════════════════════════════════════════════════════

def load_detector():
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"[ERROR] Model file not found: {MODEL_PATH}\n"
            "Download from:\n"
            "  https://storage.googleapis.com/mediapipe-models/hand_landmarker"
            "/hand_landmarker/float16/latest/hand_landmarker.task"
        )
    opts = mp_vision.HandLandmarkerOptions(
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode = mp_vision.RunningMode.VIDEO,   # ← handles any aspect ratio
        num_hands                     = 1,
        min_hand_detection_confidence = 0.5,
        min_hand_presence_confidence  = 0.5,
        min_tracking_confidence       = 0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


# ═══════════════════════════════════════════════════════════════════
# DRAWING
# ═══════════════════════════════════════════════════════════════════

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

def draw_hand(frame, lms, fw, fh):
    pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], C_BONE, 2)
    for i, (px, py) in enumerate(pts):
        col = C_INDEX if i == 8 else (C_THUMB if i == 4 else C_BONE)
        r   = 12 if i in (4, 8) else 4
        cv2.circle(frame, (px, py), r, col, -1)
    return pts[8]   # index tip (x, y)


def draw_keys(frame, buttons, hover, flashing, shift):
    for b in buttons:
        k = b["key"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]

        if   k in flashing:                bg = C_FLASH
        elif k == hover:                   bg = C_HOVER
        elif k == "SHIFT" and shift.active:bg = C_SHIFT_ON
        else:                              bg = C_KEY_BG

        cv2.rectangle(frame, (x, y), (x+w, y+h), bg,      -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), C_KEY_BOR, 1)

        if   k == "SPACE": lbl = "SPACE"
        elif k == "BACK":  lbl = "< BACK"
        elif k == "SHIFT": lbl = "^ " + shift.label
        elif shift.active: lbl = k.upper()
        else:              lbl = k.lower()

        fs = 0.50 if len(lbl) > 3 else 0.64
        fw2 = 1   if len(lbl) > 3 else 2
        tc  = C_BLACK if k in flashing else C_WHITE
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, fw2)
        cv2.putText(frame, lbl,
                    (x + (w - tw)//2, y + (h + th)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, tc, fw2)


def draw_textbox(frame, text, cam_w):
    bx, by, bw, bh = 36, 16, cam_w - 72, 74
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), C_BAR_BG, -1)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (80, 80, 80), 1)
    shown = (text[-80:] if len(text) > 80 else text) + "|"
    cv2.putText(frame, shown, (bx+12, by+50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, C_WHITE, 2)


def draw_debug(frame, fps, pdist, pinching, shift, hand_ok, cam_w, cam_h):
    y0 = cam_h - 28
    cv2.rectangle(frame, (0, y0), (cam_w, cam_h), C_BAR_BG, -1)
    filled = int((1.0 - min(pdist, 0.12) / 0.12) * 16)
    bar    = "#" * filled + "." * (16 - filled)
    pinch_col = C_FLASH if pinching else C_BAR_TXT
    msg = (f"FPS:{fps:4.1f}  |  "
           f"Pinch:{pdist:.3f}(thr{PINCH_THRESH})  [{bar}]  |  "
           f"Shift:{shift.label}  |  "
           f"Hand:{'YES' if hand_ok else 'NO '}")
    cv2.putText(frame, msg, (12, y0 + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, pinch_col, 1)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    detector = load_detector()
    cap      = open_camera(CAMERA_INDEX, CAM_W, CAM_H)

    # Read one frame to get actual dimensions
    for _ in range(10):
        ok, f = cap.read()
        if ok and f is not None:
            fh, fw = f.shape[:2]
            break
    else:
        fh, fw = CAM_H, CAM_W
    print(f"[info] frame size: {fw}x{fh}")

    buttons  = build_buttons(fw)
    shift    = ShiftState()
    typed    = ""
    t_click  = 0.0
    flashing = {}          # key → expiry time
    dropped  = 0
    MAX_DROP = 80

    # ── BUG-FIX 1: frame counter for timestamps ──────────────────────────────
    # time.time() on Windows has ~15 ms resolution; consecutive frames in the
    # same tick all get the same ms value.  detect_for_video() requires strictly
    # INCREASING timestamps or it blocks.  Using frame_count * 33 guarantees
    # 0, 33, 66, 99 … which is always strictly increasing.
    frame_count = 0

    t_prev = time.time()

    cv2.namedWindow("AI Virtual Keyboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Virtual Keyboard", fw, fh)
    print("[info] ready — show your hand to the camera and pinch to type")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            dropped += 1
            if dropped >= MAX_DROP:
                print("[ERROR] camera stopped — exiting")
                break
            time.sleep(0.03)
            continue
        dropped = 0

        frame       = cv2.flip(frame, 1)
        fh, fw      = frame.shape[:2]
        frame_count += 1

        now    = time.time()
        fps    = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now

        # ── MediaPipe ────────────────────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # BUG-FIX 1: use frame_count*33 NOT time.time()-based ms
        # BUG-FIX 3: detect_for_video (VIDEO mode) handles non-square frames
        result = detector.detect_for_video(mp_img, frame_count * 33)

        # ── Hand ─────────────────────────────────────────────────────────────
        hand_ok  = bool(result.hand_landmarks)
        pdist    = 1.0
        pinching = False
        hover    = None

        if hand_ok:
            lms          = result.hand_landmarks[0]
            ix, iy       = draw_hand(frame, lms, fw, fh)
            idx_tip      = lms[8]
            thm_tip      = lms[4]
            pdist        = math.hypot(idx_tip.x - thm_tip.x,
                                      idx_tip.y - thm_tip.y)
            pinching     = pdist < PINCH_THRESH

            for b in buttons:
                bx, by, bw, bh2 = b["x"], b["y"], b["w"], b["h"]
                if bx <= ix <= bx + bw and by <= iy <= by + bh2:
                    hover = b["key"]
                    if pinching and (now - t_click) > CLICK_COOLDOWN:
                        k           = b["key"]
                        flashing[k] = now + FLASH_SECS
                        t_click     = now
                        if k == "SPACE":
                            pyautogui.press("space")
                            typed += " "
                        elif k == "BACK":
                            pyautogui.press("backspace")
                            typed = typed[:-1]
                        elif k == "SHIFT":
                            shift.toggle()
                        else:
                            ch     = k.upper() if shift.active else k.lower()
                            pyautogui.press(ch)
                            typed += ch
                            shift.consume()
                    break

        # ── Expire flashes ───────────────────────────────────────────────────
        flashing = {k: v for k, v in flashing.items() if v > now}

        # ── Draw ─────────────────────────────────────────────────────────────
        draw_keys(frame, buttons, hover, set(flashing), shift)
        draw_textbox(frame, typed, fw)
        draw_debug(frame, fps, pdist, pinching, shift, hand_ok, fw, fh)

        cv2.imshow("AI Virtual Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\n[done] typed text: {typed}")


if __name__ == "__main__":
    main()
