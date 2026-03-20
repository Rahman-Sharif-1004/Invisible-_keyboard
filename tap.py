import cv2
import mediapipe as mp
import urllib.request
import os
import time
import threading
import collections
from pynput.keyboard import Controller, Key

# ── Download model ──────────────────────────────────────────
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmark model... please wait")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# ════════════════════════════════════════════════════════════
#  TUNING
# ════════════════════════════════════════════════════════════
TAP_THRESHOLD = 4.5
COOLDOWN      = 0.4

# ════════════════════════════════════════════════════════════
#  KEYBOARD
# ════════════════════════════════════════════════════════════
ROWS = [
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L'],
    ['Z','X','C','V','B','N','M'],
    ['SPACE','BACK']
]
FINGERTIPS = [4, 8, 12, 16, 20]

kb = Controller()
KB = {}

def init_keyboard(fw, fh):
    gap     = max(3, int(fw * 0.007))
    key_w   = int((fw - gap * 11) / 10)
    key_h   = int(fh * 0.10)
    start_x = gap
    start_y = int(fh * 0.50)
    KB.update(dict(
        gap=gap, key_w=key_w, key_h=key_h,
        start_x=start_x, start_y=start_y,
        fw=fw, fh=fh
    ))

def build_keys():
    kw  = KB['key_w']
    kh  = KB['key_h']
    gap = KB['gap']
    sx  = KB['start_x']
    sy  = KB['start_y']
    fw  = KB['fw']
    keys = []
    for ri, row in enumerate(ROWS):
        rx = sx + ri * int(kw * 0.20)
        ry = sy + ri * (kh + gap)
        for ci, letter in enumerate(row):
            if letter == 'SPACE':
                kx, kw2 = sx, int(fw * 0.50)
            elif letter == 'BACK':
                space_w = int(fw * 0.50)
                kx      = sx + space_w + gap
                kw2     = fw - sx*2 - space_w - gap
            else:
                kx, kw2 = rx + ci*(kw+gap), kw
            keys.append(dict(letter=letter, x=kx, y=ry, w=kw2, h=kh))
    return keys

# ════════════════════════════════════════════════════════════
#  UI DRAWING
# ════════════════════════════════════════════════════════════

# Color palette
C_KEY_IDLE    = (30, 30, 30)
C_KEY_HOVER   = (80, 55, 120)
C_KEY_PRESS   = (40, 180, 70)
C_KEY_SPECIAL = (45, 35, 65)
C_BORDER      = (100, 100, 100)
C_BORDER_HOV  = (160, 120, 220)
C_BORDER_PRE  = (80, 220, 120)
C_TEXT        = (230, 230, 230)
C_TEXT_DIM    = (130, 130, 130)
C_ACCENT      = (0, 210, 140)
C_BAR_BG      = (12, 12, 12)
C_HAND_0      = (220, 80, 220)
C_HAND_1      = (80, 200, 240)

def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    """Draw a rectangle with rounded corners."""
    if thickness == -1:
        cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, -1)
        cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, -1)
        cv2.circle(img, (x+r,   y+r),   r, color, -1)
        cv2.circle(img, (x+w-r, y+r),   r, color, -1)
        cv2.circle(img, (x+r,   y+h-r), r, color, -1)
        cv2.circle(img, (x+w-r, y+h-r), r, color, -1)
    else:
        cv2.rectangle(img, (x+r, y),   (x+w-r, y),   color, thickness)
        cv2.rectangle(img, (x+r, y+h), (x+w-r, y+h), color, thickness)
        cv2.rectangle(img, (x,   y+r), (x,   y+h-r), color, thickness)
        cv2.rectangle(img, (x+w, y+r), (x+w, y+h-r), color, thickness)
        cv2.ellipse(img, (x+r,   y+r),   (r,r), 180, 0, 90,  color, thickness)
        cv2.ellipse(img, (x+w-r, y+r),   (r,r), 270, 0, 90,  color, thickness)
        cv2.ellipse(img, (x+r,   y+h-r), (r,r), 90,  0, 90,  color, thickness)
        cv2.ellipse(img, (x+w-r, y+h-r), (r,r), 0,   0, 90,  color, thickness)

def draw_keyboard(frame, keys, pressed_keys, hover_keys, debug=False):
    overlay = frame.copy()
    kw  = KB['key_w']
    kh  = KB['key_h']
    r   = max(4, kh // 6)   # corner radius

    for key in keys:
        x, y, w, h = key['x'], key['y'], key['w'], key['h']
        label = key['letter']
        is_special = label in ('SPACE', 'BACK')

        # Background color
        if key in pressed_keys:
            bg  = C_KEY_PRESS
            brd = C_BORDER_PRE
        elif key in hover_keys:
            bg  = C_KEY_HOVER
            brd = C_BORDER_HOV
        elif is_special:
            bg  = C_KEY_SPECIAL
            brd = C_BORDER
        else:
            bg  = C_KEY_IDLE
            brd = C_BORDER

        # Draw key background
        draw_rounded_rect(overlay, x+1, y+1, w-2, h-2, r, bg)

        # Draw key border
        draw_rounded_rect(overlay, x+1, y+1, w-2, h-2, r, brd, 1)

        # Top highlight line (gives 3D feel)
        if key not in pressed_keys:
            highlight = (min(bg[0]+40,255), min(bg[1]+40,255), min(bg[2]+40,255))
            cv2.line(overlay, (x+r+2, y+2), (x+w-r-2, y+2), highlight, 1)

        # Label text
        if is_special:
            fs = max(0.28, kw / 200)
        else:
            fs = max(0.40, kw / 130)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2

        # Shadow text
        cv2.putText(overlay, label, (tx+1, ty+1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), 2)
        # Main text
        cv2.putText(overlay, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, C_TEXT, 1)

        # Debug mode — red outline
        if debug:
            cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,0,255), 1)

    # Blend keyboard onto frame
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

def draw_top_bar(frame, typed_text, fw, fh):
    """Text display bar at the top."""
    bar_h = 58
    # Background
    cv2.rectangle(frame, (0,0), (fw, bar_h), C_BAR_BG, -1)
    # Bottom border line
    cv2.line(frame, (0, bar_h), (fw, bar_h), (40,40,40), 1)

    # Cursor blink — on for 0.6s off for 0.4s
    cursor = "|" if int(time.time() * 1.67) % 2 == 0 else " "

    display = typed_text[-62:] if len(typed_text) > 62 else typed_text

    # Prompt symbol
    cv2.putText(frame, ">", (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_ACCENT, 2)

    # Typed text
    cv2.putText(frame, " " + display + cursor, (30, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_TEXT, 1)

def draw_bottom_bar(frame, hand_count, fps, fw, fh, debug_mode):
    """Status bar at the bottom."""
    bar_h = 30
    y0    = fh - bar_h
    cv2.rectangle(frame, (0, y0), (fw, fh), C_BAR_BG, -1)
    cv2.line(frame, (0, y0), (fw, y0), (40,40,40), 1)

    # Hand status dots
    for i in range(2):
        cx  = 18 + i * 22
        cy  = y0 + 15
        col = (C_HAND_0 if i == 0 else C_HAND_1) if i < hand_count else (50,50,50)
        cv2.circle(frame, (cx, cy), 7, col, -1)
        cv2.circle(frame, (cx, cy), 7, (80,80,80), 1)

    # Hand count text
    hand_txt = f"Hands: {hand_count}/2"
    cv2.putText(frame, hand_txt, (52, y0+19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT_DIM, 1)

    # FPS
    fps_col = C_ACCENT if fps >= 25 else (0, 140, 220) if fps >= 15 else (0,80,200)
    cv2.putText(frame, f"FPS: {fps:.0f}", (fw//2 - 30, y0+19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, fps_col, 1)

    # Shortcuts
    dbg = "D=debug off" if debug_mode else "D=debug"
    shortcuts = f"{dbg}   F=window   Q=quit"
    (sw,_),_ = cv2.getTextSize(shortcuts, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.putText(frame, shortcuts, (fw - sw - 10, y0+19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT_DIM, 1)

def draw_hand_skeleton(frame, hand_lm, fw, fh, colour):
    """Draw hand skeleton with gradient-style lines."""
    CONNS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    # Draw lines
    for s, e in CONNS:
        x1 = fw - int(hand_lm[s].x * fw)
        y1 = int(hand_lm[s].y * fh)
        x2 = fw - int(hand_lm[e].x * fw)
        y2 = int(hand_lm[e].y * fh)
        cv2.line(frame, (x1,y1), (x2,y2), colour, 1)

    # Draw joint dots
    for i, lm in enumerate(hand_lm):
        px = fw - int(lm.x * fw)
        py = int(lm.y * fh)
        r  = 4 if i == 0 else 3
        cv2.circle(frame, (px, py), r, colour, -1)
        cv2.circle(frame, (px, py), r, (200,200,200), 1)

def draw_fingertip(frame, cx, cy, colour, is_on_key):
    """Draw fingertip with glow effect when on a key."""
    if is_on_key:
        # Outer glow ring
        cv2.circle(frame, (cx,cy), 14, (colour[0]//3, colour[1]//3, colour[2]//3), -1)
        cv2.circle(frame, (cx,cy), 13, colour, 1)
    # Main dot
    cv2.circle(frame, (cx,cy), 9, colour, -1)
    cv2.circle(frame, (cx,cy), 9, (240,240,240), 1)

def draw_no_hands_hint(frame, fw, fh):
    """Show hint when no hands detected."""
    msg  = "Show your hands to the camera"
    fs   = 0.65
    (tw,th),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    tx = (fw - tw) // 2
    ty = int(fh * 0.35)
    # Shadow
    cv2.putText(frame, msg, (tx+1, ty+1),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), 2)
    # Text
    cv2.putText(frame, msg, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, fs, C_ACCENT, 2)

def get_key_at(keys, cx, cy):
    for key in keys:
        if (key['x'] <= cx <= key['x']+key['w'] and
                key['y'] <= cy <= key['y']+key['h']):
            return key
    return None

def press_key(letter):
    if letter == 'SPACE':
        kb.press(Key.space);      kb.release(Key.space)
    elif letter == 'BACK':
        kb.press(Key.backspace);  kb.release(Key.backspace)
    else:
        kb.press(letter.lower()); kb.release(letter.lower())

# ════════════════════════════════════════════════════════════
#  SMOOTHER
# ════════════════════════════════════════════════════════════
class Smoother:
    def __init__(self, window=4):
        self.buf = collections.deque(maxlen=window)
    def update(self, v):
        self.buf.append(v)
        return int(sum(self.buf) / len(self.buf))
    def reset(self):
        self.buf.clear()

# ════════════════════════════════════════════════════════════
#  TAP DETECTOR — exact working version
# ════════════════════════════════════════════════════════════
class FingerTapDetector:
    IDLE    = 0
    LIFTING = 1

    def __init__(self):
        self.smoother = Smoother(window=4)
        self.prev_y   = None
        self.last_tap = 0
        self.state    = self.IDLE

    def update(self, cy):
        now       = time.time()
        smooth_cy = self.smoother.update(cy)
        tapped    = False
        if self.prev_y is not None:
            dy = smooth_cy - self.prev_y
            if self.state == self.IDLE:
                if dy > TAP_THRESHOLD and (now - self.last_tap) > COOLDOWN:
                    tapped        = True
                    self.last_tap = now
                    self.state    = self.LIFTING
            elif self.state == self.LIFTING:
                if dy < -3:
                    self.state = self.IDLE
        self.prev_y = smooth_cy
        return tapped

    def reset(self):
        self.prev_y = None
        self.state  = self.IDLE
        self.smoother.reset()

# ════════════════════════════════════════════════════════════
#  HANDS TRACKER
# ════════════════════════════════════════════════════════════
class HandsTracker:
    def __init__(self):
        self.detectors       = [FingerTapDetector() for _ in range(10)]
        self.prev_hand_count = 0

    def update(self, hand_i, finger_i, cy):
        return self.detectors[hand_i*5+finger_i].update(cy)

    def reset_hand(self, hand_i):
        for i in range(5):
            self.detectors[hand_i*5+i].reset()

# ════════════════════════════════════════════════════════════
#  MEDIAPIPE LIVE STREAM
# ════════════════════════════════════════════════════════════
latest_result      = None
latest_result_lock = threading.Lock()

def on_result(result, output_image, timestamp_ms):
    global latest_result
    with latest_result_lock:
        latest_result = result

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    min_hand_presence_confidence=0.5,
    result_callback=on_result
)

# ════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_AUTOFOCUS,    0)

tracker     = HandsTracker()
typed_text  = ""
flash_until = {}
timestamp   = 0
keys        = []
debug_mode  = False
fps_buf     = collections.deque(maxlen=30)

HAND_COLOURS = [C_HAND_0, C_HAND_1]

cv2.namedWindow("Invisible Keyboard", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Invisible Keyboard",
                       cv2.WND_PROP_FULLSCREEN,
                       cv2.WINDOW_FULLSCREEN)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        now    = time.time()
        fps_buf.append(now)

        if not keys:
            init_keyboard(fw, fh)
            keys = build_keys()

        # Send raw frame to MediaPipe before flipping
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp += 1
        landmarker.detect_async(mp_image, timestamp)

        # Flip for display
        frame = cv2.flip(frame, 1)

        with latest_result_lock:
            result = latest_result

        hand_count = (len(result.hand_landmarks)
                      if result and result.hand_landmarks else 0)

        if hand_count < tracker.prev_hand_count:
            for hi in range(hand_count, tracker.prev_hand_count):
                tracker.reset_hand(hi)
        tracker.prev_hand_count = hand_count

        hover_keys = []

        if result and result.hand_landmarks:
            for hand_i, hand_lm in enumerate(result.hand_landmarks):
                colour = HAND_COLOURS[hand_i % 2]

                # Draw skeleton
                draw_hand_skeleton(frame, hand_lm, fw, fh, colour)

                # Process fingertips
                for finger_i, tip_id in enumerate(FINGERTIPS):
                    lm      = hand_lm[tip_id]
                    flip_cx = fw - int(lm.x * fw)
                    raw_cy  = int(lm.y * fh)

                    current_key = get_key_at(keys, flip_cx, raw_cy)
                    is_on_key   = current_key is not None

                    # Draw fingertip
                    draw_fingertip(frame, flip_cx, raw_cy, colour, is_on_key)

                    if is_on_key and current_key not in hover_keys:
                        hover_keys.append(current_key)

                    # Tap detection — exact working algorithm
                    if tracker.update(hand_i, finger_i, raw_cy):
                        if current_key:
                            press_key(current_key['letter'])
                            if current_key['letter'] == 'SPACE':
                                typed_text += ' '
                            elif current_key['letter'] == 'BACK':
                                typed_text = typed_text[:-1]
                            else:
                                typed_text += current_key['letter']
                            flash_until[id(current_key)] = now + 0.20
                            print(f"  {current_key['letter']}  "
                                  f"({flip_cx},{raw_cy})")
        else:
            # No hands — show hint
            draw_no_hands_hint(frame, fw, fh)

        pressed_keys = [k for k in keys if flash_until.get(id(k), 0) > now]
        draw_keyboard(frame, keys, pressed_keys, hover_keys, debug=debug_mode)

        # FPS calculation
        fps = (len(fps_buf)/(fps_buf[-1]-fps_buf[0]+1e-9)
               if len(fps_buf) > 1 else 0)

        draw_top_bar(frame, typed_text, fw, fh)
        draw_bottom_bar(frame, hand_count, fps, fw, fh, debug_mode)

        cv2.imshow("Invisible Keyboard", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('d'):
            debug_mode = not debug_mode
        elif k == ord('f'):
            prop = cv2.getWindowProperty(
                "Invisible Keyboard", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                "Invisible Keyboard", cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
                if prop == cv2.WINDOW_FULLSCREEN
                else cv2.WINDOW_FULLSCREEN)

cap.release()
cv2.destroyAllWindows()