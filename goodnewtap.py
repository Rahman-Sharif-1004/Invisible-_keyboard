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
#  TUNING — only change these two values
#  TAP_THRESHOLD : lower = easier tap, higher = harder tap
#  COOLDOWN      : seconds before same finger can tap again
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
    print(f"Screen {fw}x{fh} | key {key_w}x{key_h} | "
          f"keyboard y={start_y}..{start_y+4*(key_h+gap)}")

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

def draw_keyboard(frame, keys, pressed_keys, hover_keys, debug=False):
    overlay = frame.copy()
    kw = KB['key_w']
    for key in keys:
        x, y, w, h = key['x'], key['y'], key['w'], key['h']
        if key in pressed_keys:
            bg = (0, 200, 80)
        elif key in hover_keys:
            bg = (70, 50, 110)
        else:
            bg = (28, 28, 28)
        border = (0, 0, 255) if debug else (150, 150, 150)
        thick  = 2           if debug else 1
        cv2.rectangle(overlay, (x,y), (x+w,y+h), bg, -1)
        cv2.rectangle(overlay, (x,y), (x+w,y+h), border, thick)
        label = key['letter']
        fs    = (max(0.30, kw/185) if len(label) > 1
                 else max(0.42, kw/128))
        (tw,th),_ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.putText(overlay, label,
                    (x+(w-tw)//2, y+(h+th)//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (235,235,235), 2)
    cv2.addWeighted(overlay, 0.52, frame, 0.48, 0, frame)

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
#  TAP DETECTOR — the exact version that worked
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

HAND_COLOURS = [(255,0,255), (0,200,255)]

CONNS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

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

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp += 1
        landmarker.detect_async(mp_image, timestamp)

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

                for s, e in CONNS:
                    cv2.line(frame,
                        (fw-int(hand_lm[s].x*fw), int(hand_lm[s].y*fh)),
                        (fw-int(hand_lm[e].x*fw), int(hand_lm[e].y*fh)),
                        (70,180,70), 1)

                for lm in hand_lm:
                    cv2.circle(frame,
                        (fw-int(lm.x*fw), int(lm.y*fh)),
                        3, (170,170,170), -1)

                for finger_i, tip_id in enumerate(FINGERTIPS):
                    lm      = hand_lm[tip_id]
                    flip_cx = fw - int(lm.x * fw)
                    raw_cy  = int(lm.y * fh)

                    cv2.circle(frame, (flip_cx,raw_cy), 10, colour,       -1)
                    cv2.circle(frame, (flip_cx,raw_cy), 10, (255,255,255),  1)

                    current_key = get_key_at(keys, flip_cx, raw_cy)
                    if current_key and current_key not in hover_keys:
                        hover_keys.append(current_key)

                    if tracker.update(hand_i, finger_i, raw_cy):
                        if current_key:
                            press_key(current_key['letter'])
                            if current_key['letter'] == 'SPACE':
                                typed_text += ' '
                            elif current_key['letter'] == 'BACK':
                                typed_text = typed_text[:-1]
                            else:
                                typed_text += current_key['letter']
                            flash_until[id(current_key)] = now + 0.22
                            print(f"  {current_key['letter']}  "
                                  f"({flip_cx},{raw_cy})")

        pressed_keys = [k for k in keys if flash_until.get(id(k), 0) > now]
        draw_keyboard(frame, keys, pressed_keys, hover_keys, debug=debug_mode)

        # Top bar
        cv2.rectangle(frame, (0,0), (fw,54), (8,8,8), -1)
        display = typed_text[-65:] if len(typed_text) > 65 else typed_text
        cv2.putText(frame, "> " + display + "|",
                    (12,36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78, (0,230,160), 2)

        # Bottom bar
        cv2.rectangle(frame, (0,fh-28), (fw,fh), (8,8,8), -1)
        fps = (len(fps_buf)/(fps_buf[-1]-fps_buf[0]+1e-9)
               if len(fps_buf) > 1 else 0)
        dbg = "D=off" if debug_mode else "D=debug"
        cv2.putText(frame,
                    f"Hands:{hand_count}  FPS:{fps:.0f}  "
                    f"{dbg}  F=window  Q=quit",
                    (12,fh-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (110,110,110), 1)

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