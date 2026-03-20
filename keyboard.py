import cv2
import mediapipe as mp
import urllib.request
import os

# ── Download model if not present ──────────────────────────
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmark model... please wait")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# ── MediaPipe setup ─────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ── Keyboard layout ─────────────────────────────────────────
ROWS = [
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L'],
    ['Z','X','C','V','B','N','M'],
    ['SPACE']
]

KEY_W   = 52
KEY_H   = 52
GAP     = 4
START_X = 15
START_Y = 220

# All 5 fingertip landmark IDs
# 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
FINGERTIPS = [4, 8, 12, 16, 20]

# ── Build key list ──────────────────────────────────────────
def build_keys():
    keys = []
    for row_i, row in enumerate(ROWS):
        row_x = START_X + row_i * 18
        row_y = START_Y + row_i * (KEY_H + GAP)

        for col_i, letter in enumerate(row):
            if letter == 'SPACE':
                # Spacebar — full width, centred under the keyboard
                key_w = KEY_W * 8 + GAP * 7   # as wide as top row
                key_x = START_X               # starts from left edge
            else:
                key_w = KEY_W
                key_x = row_x + col_i * (KEY_W + GAP)

            keys.append({
                'letter': letter,
                'x':      key_x,
                'y':      row_y,
                'w':      key_w,
                'h':      KEY_H
            })
    return keys

# ── Draw all keys ───────────────────────────────────────────
def draw_keyboard(frame, keys, hovered_keys):
    for key in keys:
        x, y, w, h = key['x'], key['y'], key['w'], key['h']

        if key in hovered_keys:
            bg = (160, 100, 240)   # purple = hovered
        else:
            bg = (40, 40, 40)      # dark grey = idle

        cv2.rectangle(frame, (x, y), (x+w, y+h), bg, -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 180, 180), 1)

        label      = key['letter']
        font_scale = 0.40 if label == 'SPACE' else 0.62

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        cv2.putText(
            frame, label,
            (x + (w - tw) // 2, y + (h + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), 2
        )

# ── Check hover for one fingertip ───────────────────────────
def get_hover_key(keys, cx, cy):
    for key in keys:
        if (key['x'] <= cx <= key['x'] + key['w'] and
                key['y'] <= cy <= key['y'] + key['h']):
            return key
    return None

# ── Draw skeleton + all 5 fingertips for one hand ───────────
def draw_hand(frame, hand_landmarks, fw, fh, colour):
    """
    Now returns a LIST of 5 fingertip positions
    instead of just the index finger.
    """
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]

    # Skeleton lines
    for s, e in connections:
        x1 = int(hand_landmarks[s].x * fw)
        y1 = int(hand_landmarks[s].y * fh)
        x2 = int(hand_landmarks[e].x * fw)
        y2 = int(hand_landmarks[e].y * fh)
        cv2.line(frame, (x1,y1), (x2,y2), (100, 200, 100), 2)

    # All 21 joint dots
    for lm in hand_landmarks:
        cv2.circle(frame,
                   (int(lm.x * fw), int(lm.y * fh)),
                   4, (200, 200, 200), -1)

    # Draw all 5 fingertips and collect their positions
    fingertip_positions = []
    for tip_id in FINGERTIPS:
        tip = hand_landmarks[tip_id]
        cx  = int(tip.x * fw)
        cy  = int(tip.y * fh)

        # Bigger coloured dot on each fingertip
        cv2.circle(frame, (cx, cy), 13, colour, -1)

        fingertip_positions.append((cx, cy))

    return fingertip_positions   # returns list of 5 (cx,cy) tuples

# ── Main loop ────────────────────────────────────────────────
keys = build_keys()
cap  = cv2.VideoCapture(0)

HAND_COLOURS = [
    (255, 0, 255),    # magenta — first hand
    (0, 200, 255),    # yellow  — second hand
]

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        hovered_keys = []

        if result.hand_landmarks:
            fh, fw, _ = frame.shape

            for hand_i, hand_landmarks in enumerate(result.hand_landmarks):
                colour = HAND_COLOURS[hand_i % 2]

                # Get all 5 fingertip positions for this hand
                fingertip_positions = draw_hand(
                    frame, hand_landmarks, fw, fh, colour
                )

                # Check every fingertip against every key
                for cx, cy in fingertip_positions:
                    hkey = get_hover_key(keys, cx, cy)
                    if hkey and hkey not in hovered_keys:
                        hovered_keys.append(hkey)

        draw_keyboard(frame, keys, hovered_keys)

        # Show hand count
        hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0
        cv2.putText(frame,
                    f"Hands: {hand_count}  |  Fingers on keys: {len(hovered_keys)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 230, 160), 2)

        cv2.imshow("Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()