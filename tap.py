import cv2
import mediapipe as mp
import urllib.request
import os
import time
import threading
import collections
from pynput.keyboard import Controller, Key
import numpy as np

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
#  COLOR PALETTE — Deep Blue Theme
# ════════════════════════════════════════════════════════════
C_KEY_IDLE    = (50,  28,  8)
C_KEY_HOVER   = (110, 60,  12)
C_KEY_PRESS   = (140, 60,  10)
C_KEY_SPECIAL = (70,  38,  12)
C_KEY_NUM     = (60,  33,  10)
C_KEY_CAPS_ON = (20,  140, 170)
C_BORDER      = (100, 65,  25)
C_BORDER_HOV  = (160, 110, 45)
C_BORDER_PRE  = (200, 100, 20)
C_BORDER_CAPS = (30,  200, 220)
C_TEXT        = (230, 218, 195)
C_TEXT_BRIGHT = (255, 248, 225)
C_TEXT_DIM    = (110, 95,  65)
C_ACCENT      = (0,   195, 255)
C_ACCENT2     = (0,   255, 170)
C_BAR_BG      = (12,  6,   2)
C_HAND_0      = (170, 50,  210)
C_HAND_1      = (10,  185, 255)
C_TITLE       = (0,   205, 255)
C_CAPS_ACTIVE = (0,   230, 195)

caps_on = False

# ════════════════════════════════════════════════════════════
#  KEYBOARD LAYOUT
# ════════════════════════════════════════════════════════════
MAIN_ROWS = [
    ['`','1','2','3','4','5','6','7','8','9','0','-','=','BACK'],
    ['Q','W','E','R','T','Y','U','I','O','P','[',']','\\'],
    ['A','S','D','F','G','H','J','K','L',';',"'",'ENTER'],
    ['Z','X','C','V','B','N','M',',','.','/', 'CAPS'],
    ['SPACE']
]
NUM_ROWS = [
    ['7','8','9'],
    ['4','5','6'],
    ['1','2','3'],
    ['0',  '.'],
]
FINGERTIPS = [4, 8, 12, 16, 20]

kb = Controller()
KB = {}

# ════════════════════════════════════════════════════════════
#  KEYBOARD SIZING
# ════════════════════════════════════════════════════════════
def init_keyboard(fw, fh):
    margin     = int(fw * 0.005)
    gap        = max(2, int(fw * 0.004))
    usable     = fw - 2*margin
    nk_cols    = 4
    nk_w_total = int(usable * 0.205)
    nk_gap     = gap
    nk_w       = int((nk_w_total - nk_gap*nk_cols) / nk_cols)
    divider    = int(fw * 0.012)
    mk_w_total = usable - nk_w_total - divider
    key_w      = int((mk_w_total - gap*15) / 14)
    key_h      = int(fh * 0.080)
    start_x    = margin
    start_y    = int(fh * 0.50)
    nk_x       = margin + mk_w_total + divider
    nk_y       = start_y
    nk_h       = key_h
    KB.update(dict(
        gap=gap, key_w=key_w, key_h=key_h,
        start_x=start_x, start_y=start_y,
        nk_w=nk_w, nk_h=nk_h, nk_gap=nk_gap,
        nk_x=nk_x, nk_y=nk_y,
        mk_w_total=mk_w_total,
        fw=fw, fh=fh, margin=margin
    ))
    print(f"Screen {fw}x{fh} | key_w={key_w} key_h={key_h}")

def build_keys():
    kw  = KB['key_w']
    kh  = KB['key_h']
    gap = KB['gap']
    sx  = KB['start_x']
    sy  = KB['start_y']
    mkw = KB['mk_w_total']
    nkw = KB['nk_w']
    nkh = KB['nk_h']
    ng  = KB['nk_gap']
    nx  = KB['nk_x']
    ny  = KB['nk_y']
    keys = []

    indents = [0, int(kw*0.5), int(kw*0.75),
               int(kw*1.25),   int(kw*0.75)]

    for ri, row in enumerate(MAIN_ROWS):
        ry = sy + ri*(kh+gap)
        rx = sx + indents[ri]
        for ci, letter in enumerate(row):
            if letter == 'BACK':
                used = (len(row)-1)*(kw+gap)
                kw2  = sx + mkw - (rx + used)
                kx   = rx + used
            elif letter == 'ENTER':
                used = (len(row)-1)*(kw+gap)
                kw2  = sx + mkw - (rx + used)
                kx   = rx + used
            elif letter == 'CAPS':
                kw2 = sx + mkw - (rx + ci*(kw+gap))
                kx  = rx + ci*(kw+gap)
            elif letter == 'SPACE':
                kw2 = int(kw * 6.5)
                kx  = sx + indents[4]
            else:
                kw2 = kw
                kx  = rx + ci*(kw+gap)
            if kw2 < 8:
                kw2 = kw
            keys.append(dict(letter=letter, x=kx, y=ry,
                             w=kw2, h=kh, kind='main'))

    for ri, row in enumerate(NUM_ROWS):
        ry = ny + ri*(nkh+ng)
        for ci, letter in enumerate(row):
            kx = nx + ci*(nkw+ng)
            keys.append(dict(letter=letter, x=kx, y=ry,
                             w=nkw, h=nkh, kind='num'))

    rc_x = nx + 3*(nkw+ng)
    keys.append(dict(letter='ENTER', x=rc_x, y=ny,
                     w=nkw, h=nkh*2+ng, kind='num'))
    keys.append(dict(letter='+',     x=rc_x, y=ny+2*(nkh+ng),
                     w=nkw, h=nkh,    kind='num'))
    keys.append(dict(letter='BACK',  x=rc_x, y=ny+3*(nkh+ng),
                     w=nkw, h=nkh,    kind='num'))

    return keys

# ════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ════════════════════════════════════════════════════════════
def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    r = max(1, min(r, w//2-1, h//2-1))
    if thickness == -1:
        cv2.rectangle(img,(x+r,y),(x+w-r,y+h),color,-1)
        cv2.rectangle(img,(x,y+r),(x+w,y+h-r),color,-1)
        for cx2,cy2 in [(x+r,y+r),(x+w-r,y+r),
                        (x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(img,(cx2,cy2),r,color,-1)
    else:
        cv2.line(img,(x+r,y),(x+w-r,y),color,thickness)
        cv2.line(img,(x+r,y+h),(x+w-r,y+h),color,thickness)
        cv2.line(img,(x,y+r),(x,y+h-r),color,thickness)
        cv2.line(img,(x+w,y+r),(x+w,y+h-r),color,thickness)
        cv2.ellipse(img,(x+r,  y+r),  (r,r),180,0,90,color,thickness)
        cv2.ellipse(img,(x+w-r,y+r),  (r,r),270,0,90,color,thickness)
        cv2.ellipse(img,(x+r,  y+h-r),(r,r),90, 0,90,color,thickness)
        cv2.ellipse(img,(x+w-r,y+h-r),(r,r),0,  0,90,color,thickness)

def put_text_c(img, text, cx, cy, scale, color, thickness=1):
    (tw,th),_ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    tx = cx - tw//2
    ty = cy + th//2
    cv2.putText(img,text,(tx+1,ty+1),
                cv2.FONT_HERSHEY_DUPLEX,scale,(0,0,0),thickness+1)
    cv2.putText(img,text,(tx,ty),
                cv2.FONT_HERSHEY_DUPLEX,scale,color,thickness)

def draw_keyboard(frame, keys, pressed_keys, hover_keys, debug=False):
    overlay = frame.copy()
    kh = KB['key_h']
    r  = max(4, kh//6)

    for key in keys:
        x,y,w,h = key['x'],key['y'],key['w'],key['h']
        label   = key['letter']
        kind    = key.get('kind','main')
        special = label in ('BACK','ENTER','SPACE','CAPS','+')

        if key in pressed_keys:
            bg,brd = C_KEY_PRESS, C_BORDER_PRE
        elif label=='CAPS' and caps_on:
            bg,brd = C_KEY_CAPS_ON, C_BORDER_CAPS
        elif key in hover_keys:
            bg,brd = C_KEY_HOVER, C_BORDER_HOV
        elif kind=='num':
            bg,brd = C_KEY_NUM, C_BORDER
        elif special:
            bg,brd = C_KEY_SPECIAL, C_BORDER
        else:
            bg,brd = C_KEY_IDLE, C_BORDER

        draw_rounded_rect(overlay,x+1,y+1,w-2,h-2,r,bg)
        hl = tuple(min(c+50,255) for c in bg)
        cv2.line(overlay,(x+r+2,y+3),(x+w-r-2,y+3),hl,1)
        draw_rounded_rect(overlay,x+1,y+1,w-2,h-2,r,brd,1)

        if key in hover_keys or key in pressed_keys:
            inner = tuple(min(c+20,255) for c in brd)
            draw_rounded_rect(overlay,x+3,y+3,w-6,h-6,
                              max(1,r-2),inner,1)

        disp = label
        if label == 'BACK': disp = '<<'

        if kind=='num' and label not in ('ENTER','BACK','+'):
            fs = max(0.32, KB['nk_w']/88)
        elif special:
            fs = max(0.22, KB['key_w']/188)
        else:
            fs = max(0.24, KB['key_w']/162)

        if label=='CAPS' and caps_on:
            tc = (210,255,255)
        elif key in pressed_keys:
            tc = (255,230,180)
        elif special:
            tc = C_ACCENT
        else:
            tc = C_TEXT_BRIGHT

        put_text_c(overlay,disp,x+w//2,y+h//2,fs,tc)

        if label=='CAPS' and caps_on:
            cv2.circle(overlay,(x+w-7,y+7),3,(0,255,210),-1)

        if debug:
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,0,255),1)

    put_text_c(overlay,"NUM",
               KB['nk_x']+KB['nk_w'], KB['nk_y']-10,
               0.30, C_TEXT_DIM)

    cv2.addWeighted(overlay,0.65,frame,0.35,0,frame)

def draw_top_bar(frame, typed_text, fw, fh):
    bar_h = 64
    cv2.rectangle(frame,(0,0),(fw,bar_h),C_BAR_BG,-1)
    cv2.line(frame,(0,0),(fw,0),C_ACCENT,2)

    cv2.putText(frame,"INVISIBLE  KEYBOARD",(13,21),
                cv2.FONT_HERSHEY_DUPLEX,0.52,(0,60,80),2)
    cv2.putText(frame,"INVISIBLE  KEYBOARD",(12,20),
                cv2.FONT_HERSHEY_DUPLEX,0.52,C_TITLE,1)

    if caps_on:
        draw_rounded_rect(frame,fw-85,5,75,16,4,(0,50,45),-1)
        draw_rounded_rect(frame,fw-85,5,75,16,4,C_CAPS_ACTIVE,1)
        cv2.putText(frame,"CAPS ON",(fw-80,17),
                    cv2.FONT_HERSHEY_DUPLEX,0.30,C_CAPS_ACTIVE,1)

    fx,fy,fww,fhh = 8,26,fw-16,28
    draw_rounded_rect(frame,fx,fy,fww,fhh,5,(22,12,5),-1)
    draw_rounded_rect(frame,fx,fy,fww,fhh,5,
                      (C_ACCENT[0]//5,C_ACCENT[1]//5,C_ACCENT[2]//5),1)
    cv2.putText(frame,">",(fx+7,fy+20),
                cv2.FONT_HERSHEY_DUPLEX,0.50,C_ACCENT,1)
    cursor  = "|" if int(time.time()*2)%2==0 else " "
    display = typed_text[-70:] if len(typed_text)>70 else typed_text
    cv2.putText(frame," "+display+cursor,
                (fx+23,fy+20),
                cv2.FONT_HERSHEY_DUPLEX,0.42,C_TEXT_BRIGHT,1)

def draw_bottom_bar(frame,hand_count,fps,fw,fh,debug_mode):
    bh = 24
    y0 = fh-bh
    cv2.rectangle(frame,(0,y0),(fw,fh),C_BAR_BG,-1)
    cv2.line(frame,(0,fh-1),(fw,fh-1),C_ACCENT,1)

    for i in range(2):
        cx2 = 13+i*20
        cy2 = y0+12
        col = (C_HAND_0 if i==0 else C_HAND_1) if i<hand_count else (35,35,35)
        cv2.circle(frame,(cx2,cy2),6,col,-1)
        if i<hand_count:
            cv2.circle(frame,(cx2,cy2),7,col,1)

    cv2.putText(frame,f"H:{hand_count}/2",
                (46,y0+16),
                cv2.FONT_HERSHEY_DUPLEX,0.30,C_TEXT_DIM,1)

    fps_col = C_ACCENT2 if fps>=25 else (0,150,255) if fps>=15 else (0,80,200)
    cv2.putText(frame,f"FPS {fps:.0f}",
                (fw//2-22,y0+16),
                cv2.FONT_HERSHEY_DUPLEX,0.30,fps_col,1)

    dbg  = "D=DBG OFF" if debug_mode else "D=DEBUG"
    info = f"{dbg}   F=WIN   Q=QUIT"
    (sw,_),_ = cv2.getTextSize(info,cv2.FONT_HERSHEY_DUPLEX,0.26,1)
    cv2.putText(frame,info,(fw-sw-8,y0+16),
                cv2.FONT_HERSHEY_DUPLEX,0.26,C_TEXT_DIM,1)

def draw_hand_skeleton(frame, hand_lm, fw, fh, colour):
    CONNS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    dim = tuple(max(0,c-80) for c in colour)
    for s,e in CONNS:
        x1=fw-int(hand_lm[s].x*fw); y1=int(hand_lm[s].y*fh)
        x2=fw-int(hand_lm[e].x*fw); y2=int(hand_lm[e].y*fh)
        cv2.line(frame,(x1,y1),(x2,y2),dim,2)
        cv2.line(frame,(x1,y1),(x2,y2),colour,1)
    for i,lm in enumerate(hand_lm):
        px=fw-int(lm.x*fw); py=int(lm.y*fh)
        rv = 4 if i==0 else 2
        cv2.circle(frame,(px,py),rv+1,(0,0,0),-1)
        cv2.circle(frame,(px,py),rv,colour,-1)

def draw_fingertip(frame, cx, cy, colour, is_on_key):
    if is_on_key:
        for rad,alpha in [(17,25),(13,55),(10,90)]:
            ov = frame.copy()
            cv2.circle(ov,(cx,cy),rad,colour,-1)
            cv2.addWeighted(ov,alpha/255,frame,1-alpha/255,0,frame)
    cv2.circle(frame,(cx,cy),8,colour,-1)
    cv2.circle(frame,(cx,cy),8,(255,255,255),1)
    bright = tuple(min(c+90,255) for c in colour)
    cv2.circle(frame,(cx,cy),3,bright,-1)

def draw_no_hands_hint(frame, fw, fh):
    alpha = int(180+70*abs(np.sin(time.time()*1.5)))
    col   = (0,alpha,min(255,alpha+40))
    msg   = "Show your hands to the camera"
    (tw,_),_ = cv2.getTextSize(msg,cv2.FONT_HERSHEY_DUPLEX,0.60,1)
    tx = (fw-tw)//2
    ty = int(fh*0.27)
    cv2.putText(frame,msg,(tx+1,ty+1),
                cv2.FONT_HERSHEY_DUPLEX,0.60,(0,0,0),2)
    cv2.putText(frame,msg,(tx,ty),
                cv2.FONT_HERSHEY_DUPLEX,0.60,col,1)
    sub   = "Place hands over the keyboard area below"
    (sw,_),_ = cv2.getTextSize(sub,cv2.FONT_HERSHEY_DUPLEX,0.33,1)
    cv2.putText(frame,sub,((fw-sw)//2,ty+26),
                cv2.FONT_HERSHEY_DUPLEX,0.33,C_TEXT_DIM,1)

def get_key_at(keys, cx, cy):
    for key in keys:
        if (key['x']<=cx<=key['x']+key['w'] and
                key['y']<=cy<=key['y']+key['h']):
            return key
    return None

def get_typed_char(letter):
    if len(letter)==1 and letter.isalpha():
        return letter.upper() if caps_on else letter.lower()
    return letter

def press_key(letter):
    global caps_on
    if letter=='SPACE':
        kb.press(Key.space);      kb.release(Key.space)
    elif letter=='BACK':
        kb.press(Key.backspace);  kb.release(Key.backspace)
    elif letter=='ENTER':
        kb.press(Key.enter);      kb.release(Key.enter)
    elif letter=='CAPS':
        caps_on = not caps_on
        kb.press(Key.caps_lock);  kb.release(Key.caps_lock)
    elif letter=='+':
        kb.press('+');            kb.release('+')
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
        return int(sum(self.buf)/len(self.buf))
    def reset(self):
        self.buf.clear()

# ════════════════════════════════════════════════════════════
#  TAP DETECTOR
# ════════════════════════════════════════════════════════════
class FingerTapDetector:
    IDLE=0; LIFTING=1
    def __init__(self):
        self.smoother=Smoother(window=4)
        self.prev_y=None; self.last_tap=0; self.state=self.IDLE
    def update(self, cy):
        now=time.time()
        smooth_cy=self.smoother.update(cy)
        tapped=False
        if self.prev_y is not None:
            dy=smooth_cy-self.prev_y
            if self.state==self.IDLE:
                if dy>TAP_THRESHOLD and (now-self.last_tap)>COOLDOWN:
                    tapped=True; self.last_tap=now; self.state=self.LIFTING
            elif self.state==self.LIFTING:
                if dy<-3: self.state=self.IDLE
        self.prev_y=smooth_cy
        return tapped
    def reset(self):
        self.prev_y=None; self.state=self.IDLE; self.smoother.reset()

# ════════════════════════════════════════════════════════════
#  HANDS TRACKER
# ════════════════════════════════════════════════════════════
class HandsTracker:
    def __init__(self):
        self.detectors=[FingerTapDetector() for _ in range(10)]
        self.prev_hand_count=0
    def update(self,hand_i,finger_i,cy):
        return self.detectors[hand_i*5+finger_i].update(cy)
    def reset_hand(self,hand_i):
        for i in range(5): self.detectors[hand_i*5+i].reset()

# ════════════════════════════════════════════════════════════
#  MEDIAPIPE
# ════════════════════════════════════════════════════════════
latest_result=None
latest_result_lock=threading.Lock()

def on_result(result,output_image,timestamp_ms):
    global latest_result
    with latest_result_lock:
        latest_result=result

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
#  MAIN
# ════════════════════════════════════════════════════════════
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_AUTOFOCUS,    0)

tracker    = HandsTracker()
typed_text = ""
flash_until= {}
timestamp  = 0
keys       = []
debug_mode = False
fps_buf    = collections.deque(maxlen=30)
HAND_COLOURS = [C_HAND_0, C_HAND_1]
WIN_NAME   = "Invisible Keyboard"

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN_NAME,
                       cv2.WND_PROP_FULLSCREEN,
                       cv2.WINDOW_FULLSCREEN)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret,frame=cap.read()
        if not ret: break

        fh,fw=frame.shape[:2]
        now=time.time()
        fps_buf.append(now)

        if not keys:
            init_keyboard(fw,fh)
            keys=build_keys()

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        timestamp+=1
        landmarker.detect_async(mp_image,timestamp)

        frame=cv2.flip(frame,1)

        with latest_result_lock:
            result=latest_result

        hand_count=(len(result.hand_landmarks)
                    if result and result.hand_landmarks else 0)

        if hand_count<tracker.prev_hand_count:
            for hi in range(hand_count,tracker.prev_hand_count):
                tracker.reset_hand(hi)
        tracker.prev_hand_count=hand_count

        hover_keys=[]

        if result and result.hand_landmarks:
            for hand_i,hand_lm in enumerate(result.hand_landmarks):
                colour=HAND_COLOURS[hand_i%2]
                draw_hand_skeleton(frame,hand_lm,fw,fh,colour)
                for finger_i,tip_id in enumerate(FINGERTIPS):
                    lm=hand_lm[tip_id]
                    flip_cx=fw-int(lm.x*fw)
                    raw_cy=int(lm.y*fh)
                    current_key=get_key_at(keys,flip_cx,raw_cy)
                    is_on_key=current_key is not None
                    draw_fingertip(frame,flip_cx,raw_cy,colour,is_on_key)
                    if is_on_key and current_key not in hover_keys:
                        hover_keys.append(current_key)
                    if tracker.update(hand_i,finger_i,raw_cy):
                        if current_key:
                            press_key(current_key['letter'])
                            if current_key['letter']=='SPACE':
                                typed_text+=' '
                            elif current_key['letter']=='BACK':
                                typed_text=typed_text[:-1]
                            elif current_key['letter']=='ENTER':
                                typed_text+='\n'
                            elif current_key['letter']=='CAPS':
                                pass
                            else:
                                typed_text+=get_typed_char(
                                    current_key['letter'])
                            flash_until[id(current_key)]=now+0.20
                            print(f"  {current_key['letter']} "
                                  f"({flip_cx},{raw_cy})")
        else:
            draw_no_hands_hint(frame,fw,fh)

        pressed_keys=[k for k in keys if flash_until.get(id(k),0)>now]
        draw_keyboard(frame,keys,pressed_keys,hover_keys,debug=debug_mode)

        fps=(len(fps_buf)/(fps_buf[-1]-fps_buf[0]+1e-9)
             if len(fps_buf)>1 else 0)

        draw_top_bar(frame,typed_text,fw,fh)
        draw_bottom_bar(frame,hand_count,fps,fw,fh,debug_mode)

        cv2.imshow(WIN_NAME,frame)

        k=cv2.waitKey(1)&0xFF
        if k==ord('q'):
            break
        elif k==ord('d'):
            debug_mode=not debug_mode
        elif k==ord('f'):
            prop=cv2.getWindowProperty(
                WIN_NAME,cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                WIN_NAME,cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
                if prop==cv2.WINDOW_FULLSCREEN
                else cv2.WINDOW_FULLSCREEN)

cap.release()
cv2.destroyAllWindows()