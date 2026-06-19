# -*- coding: utf-8 -*-
# ============================================================
#  🎬 ライブデモページ（AI画像認識 体験）
#  最新のStreamlit機能を活用した改善版
# ============================================================

import time
import math
from collections import deque, Counter

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# set_page_config は最初のStreamlitコマンドである必要があるため、ここで呼ぶ
st.set_page_config(page_title="🎬 ライブデモ", layout="wide")

# ============================================================
# 設定
# ============================================================
SUNGLASSES_PATH = "glasses/glasses1.png"
JP_FONT_PATH = "fonts/NotoSansJP-Regular.ttf"
EMOJI_FONT_PATH = "fonts/NotoColorEmoji.ttf"

CAM_WIDTH = 960
CAM_HEIGHT = 540
INFER_SCALE = 0.5

SUNG_WIDTH_FACTOR = 2.2
MAX_FACES = 2
MAX_HANDS = 2

FINGER_EXT_MARGIN = 0.12
THUMB_OPEN_RATIO = 0.50
GESTURE_SMOOTH_N = 5
GESTURE_MIN_VOTES = 3

GESTURE_INFO = {
    "PEACE": {"label": "ピース！", "color": (255, 120, 60)},
    "PA": {"label": "パー！", "color": (60, 200, 255)},
    "GU": {"label": "グー！", "color": (80, 80, 255)},
    "GOOD": {"label": "いいね！ GOOD", "color": (60, 220, 90)},
    "FOX": {"label": "きつね！", "color": (30, 150, 255)},
}
EMOJI_MAP = {"PEACE": "✌", "PA": "✋", "GU": "✊", "GOOD": "👍", "FOX": "🦊"}

try:
    import sounddevice as sd
    _AUDIO_OK = True
except Exception:
    _AUDIO_OK = False

SAMPLE_RATE = 44100

# ============================================================
# ユーティリティ関数
# ============================================================

def _make_tone(freqs, dur=0.18, volume=0.3):
    """指定した周波数の短い音を作る"""
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), False)
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    wave = wave / (np.max(np.abs(wave)) + 1e-9)
    fade = int(SAMPLE_RATE * 0.012)
    env = np.ones_like(wave)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (wave * env * volume).astype(np.float32)

GESTURE_SOUNDS = {
    "GU": _make_tone([196, 247]),
    "PA": _make_tone([523, 659, 784]),
    "PEACE": _make_tone([440, 554, 659]),
} if _AUDIO_OK else {}

def play_gesture_sound(gesture, enabled):
    """ジェスチャーに対応した効果音を鳴らす"""
    if enabled and _AUDIO_OK and gesture in GESTURE_SOUNDS:
        try:
            sd.play(GESTURE_SOUNDS[gesture], SAMPLE_RATE)
        except Exception:
            pass

# ============================================================
# アセット読み込み
# ============================================================

@st.cache_resource
def load_assets():
    """サングラス画像とフォントを読み込む"""
    sunglasses = cv2.imread(SUNGLASSES_PATH, cv2.IMREAD_UNCHANGED)
    if sunglasses is not None and sunglasses.shape[2] == 3:
        h, w = sunglasses.shape[:2]
        alpha = np.full((h, w, 1), 255, dtype=sunglasses.dtype)
        sunglasses = np.concatenate([sunglasses, alpha], axis=2)

    fonts = {}
    for name, size in [("xs", 20), ("s", 28), ("m", 40), ("l", 64), ("xl", 96)]:
        fonts[name] = ImageFont.truetype(JP_FONT_PATH, size)

    emoji_font = None
    emoji_ok = False
    try:
        ef = ImageFont.truetype(EMOJI_FONT_PATH, 109)
        test = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(test).text((10, 10), "👍", font=ef, embedded_color=True)
        if np.array(test)[:, :, 3].any():
            emoji_font = ef
            emoji_ok = True
    except Exception:
        emoji_ok = False

    return sunglasses, fonts, emoji_font, emoji_ok

# ============================================================
# 描画ヘルパー
# ============================================================

def overlay_rgba(background, overlay, x, y):
    """透明付き画像を背景に合成"""
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + ow), min(bh, y + oh)
    if x1 >= x2 or y1 >= y2:
        return background

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    roi = background[y1:y2, x1:x2].astype(np.float32)
    ov = overlay[oy1:oy2, ox1:ox2].astype(np.float32)
    alpha = (ov[:, :, 3:4]) / 255.0
    blended = alpha * ov[:, :, :3] + (1 - alpha) * roi
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background

_EMOJI_CACHE = {}

def get_emoji_bgra(emoji_font, emoji_char, size):
    """絵文字をBGRA画像にして返す"""
    key = (emoji_char, size)
    if key not in _EMOJI_CACHE:
        patch = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
        ImageDraw.Draw(patch).text((2, 2), emoji_char, font=emoji_font, embedded_color=True)
        bbox = patch.getbbox()
        if bbox:
            patch = patch.crop(bbox)
        patch = patch.resize((size, size), Image.LANCZOS)
        _EMOJI_CACHE[key] = cv2.cvtColor(np.array(patch), cv2.COLOR_RGBA2BGRA)
    return _EMOJI_CACHE[key]

def render_text_layer(frame, text_jobs, fonts):
    """日本語テキストをまとめて描く"""
    if not text_jobs:
        return frame
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, pos, fkey, rgb, stroke in text_jobs:
        draw.text(pos, text, font=fonts[fkey], fill=rgb,
                  stroke_width=stroke, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def render_emoji_layer(frame, emoji_jobs, emoji_font, emoji_ok):
    """絵文字をまとめて貼り付ける"""
    if not emoji_ok:
        return frame
    for char, (x, y), size in emoji_jobs:
        patch = get_emoji_bgra(emoji_font, char, size)
        frame = overlay_rgba(frame, patch, x, y)
    return frame

# ============================================================
# ジェスチャー判定
# ============================================================

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def _fingers_extended(lm, hand_size):
    """親指以外の指が伸びているか判定"""
    wrist = lm[0]
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    result = []
    for tip, pip in zip(finger_tips, finger_pips):
        diff = (_dist(lm[tip], wrist) - _dist(lm[pip], wrist)) / hand_size
        result.append(diff > FINGER_EXT_MARGIN)
    return result

def classify_gesture(hand_landmarks):
    """手のランドマークからジェスチャーを判定"""
    lm = hand_landmarks.landmark
    wrist = lm[0]
    hand_size = _dist(wrist, lm[9]) + 1e-6

    idx, mid, ring, pinky = _fingers_extended(lm, hand_size)
    n_ext = sum([idx, mid, ring, pinky])

    thumb_open = (_dist(lm[4], lm[2]) / hand_size) > THUMB_OPEN_RATIO
    thumb_up = (lm[4].y < lm[2].y - 0.04) and (lm[4].y < wrist.y)

    if idx and mid and ring and pinky and thumb_open:
        return "PA"
    if idx and pinky and (not mid) and (not ring):
        return "FOX"
    if idx and mid and (not ring) and (not pinky):
        return "PEACE"
    if n_ext == 0 and thumb_open and thumb_up:
        return "GOOD"
    if n_ext == 0:
        return "GU"
    return "UNKNOWN"

# ============================================================
# エフェクト描画
# ============================================================

def add_gesture_effect(frame, gesture, center, t, text_jobs, emoji_jobs):
    """ジェスチャーエフェクトを描く"""
    info = GESTURE_INFO[gesture]
    color = info["color"]
    cx, cy = center

    phase = (t * 1.5) % 1.0
    for k in range(3):
        r = int(40 + ((phase + k / 3.0) % 1.0) * 90)
        cv2.circle(frame, (cx, cy), r, color, 3, cv2.LINE_AA)

    for k in range(6):
        ang = t * 2 + k * (2 * math.pi / 6)
        sx = int(cx + math.cos(ang) * 75)
        sy = int(cy + math.sin(ang) * 75)
        cv2.drawMarker(frame, (sx, sy), color, cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)

    rgb = (color[2], color[1], color[0])
    text_jobs.append((info["label"], (cx - 90, cy - 150), "l", rgb, 3))
    if gesture in EMOJI_MAP:
        emoji_jobs.append((EMOJI_MAP[gesture], (cx + 95, cy - 150), 70))

def draw_confetti(frame, t):
    """紙吹雪を描く"""
    ih, iw = frame.shape[:2]
    palette = [(60, 200, 255), (80, 80, 255), (60, 220, 90),
               (255, 120, 60), (255, 80, 200), (60, 255, 255)]
    for i in range(48):
        x = (i * 137) % iw
        speed = 160 + (i * 53) % 180
        y = int((t * speed + i * 71) % (ih + 40)) - 20
        col = palette[i % len(palette)]
        cv2.rectangle(frame, (x, y), (x + 8, y + 14), col, -1)

def draw_heart(frame, center, size, color=(80, 80, 255)):
    """ハートマークを描く"""
    cx, cy = center
    r = max(6, size // 4)
    cv2.circle(frame, (cx - r, cy - r // 2), r, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + r, cy - r // 2), r, color, -1, cv2.LINE_AA)
    pts = np.array([[cx - 2 * r, cy - r // 4],
                    [cx + 2 * r, cy - r // 4],
                    [cx, cy + 2 * r]], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts, color, cv2.LINE_AA)

# ============================================================
# 顔の飾り
# ============================================================

_CATEAR_CACHE = {}
CATEAR_BASE_W = 400
CATEAR_BASE_H = 300

# 猫耳の色（RGBA）。グレーの子猫をイメージ
CAT_FUR      = ( 64,  58,  70, 255)   # 毛（濃いチャコール）
CAT_FUR_HI   = ( 98,  90, 104, 255)   # 内側の明るい毛（立体感）
CAT_FUR_LINE = ( 36,  32,  42, 255)   # 輪郭線
CAT_PINK     = (238, 150, 190, 255)   # 内耳ピンク
CAT_PINK_LT  = (250, 206, 222, 255)   # 内耳の明るい中心
CAT_TUFT     = (246, 240, 236, 255)   # ふさふさの差し毛（クリーム）

def _quad_bezier(p0, c, p1, n=16):
    """ベジェ曲線をサンプリング"""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t ** 2 * p1[1]
        pts.append((x, y))
    return pts

def _ear_outline(cx, base_y, w, h, lean=0.0):
    """先のとがった猫耳の輪郭を返す。
    両サイドはふっくら膨らませ、先端だけ少し丸めて自然な三角形にする。"""
    tipx = cx + lean
    tip   = (tipx, base_y - h)                       # とがった先端
    tip_l = (tipx - w * 0.10, base_y - h * 0.90)     # 先端の少し下（丸み用）
    tip_r = (tipx + w * 0.10, base_y - h * 0.90)
    bl = (cx - w * 0.5, base_y)                       # 根本 左
    br = (cx + w * 0.5, base_y)                       # 根本 右
    ctrl_r = (cx + w * 0.58, base_y - h * 0.40)       # 右サイドの膨らみ
    ctrl_l = (cx - w * 0.58, base_y - h * 0.40)       # 左サイドの膨らみ
    return (_quad_bezier(br, ctrl_r, tip_r, n=20)
            + _quad_bezier(tip_r, tip, tip_l, n=6)
            + _quad_bezier(tip_l, ctrl_l, bl, n=20))

def build_cat_ears(width):
    """かわいい猫耳（両耳）をBGRA画像で作る。一度作ったら覚えておく。"""
    width = max(40, int(round(width / 8) * 8))
    if width in _CATEAR_CACHE:
        return _CATEAR_CACHE[width]

    SS = 4                                            # 4倍で描いて縮小＝なめらかな輪郭
    W, H = CATEAR_BASE_W * SS, CATEAR_BASE_H * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    ear_w, ear_h = W * 0.40, H * 0.74                 # 横幅をもたせた可愛い猫耳
    base_y = H * 0.95
    for side in (-1, 1):                              # 左耳 / 右耳
        cx = W / 2 + side * W * 0.205
        lean = side * ear_w * 0.16                    # 先端をほんの少し外側へ＝猫らしい

        # 1) 毛（濃い色）のベース
        outer = _ear_outline(cx, base_y, ear_w, ear_h, lean)
        d.polygon(outer, fill=CAT_FUR)
        # 2) 内側を一回り小さく明るい毛で塗り、外周に濃い「ふち」を残す
        hi = _ear_outline(cx, base_y - ear_h * 0.02, ear_w * 0.84, ear_h * 0.90, lean)
        d.polygon(hi, fill=CAT_FUR_HI)
        # 3) 輪郭線でくっきりさせる
        d.line(outer + [outer[0]], fill=CAT_FUR_LINE, width=int(SS * 3), joint="curve")
        # 4) 内耳ピンク（小さめに、少し上へ）
        inner = _ear_outline(cx, base_y - ear_h * 0.05, ear_w * 0.52, ear_h * 0.62, lean)
        d.polygon(inner, fill=CAT_PINK)
        inner2 = _ear_outline(cx, base_y - ear_h * 0.05, ear_w * 0.30, ear_h * 0.44, lean)
        d.polygon(inner2, fill=CAT_PINK_LT)
        # 5) 根本のふさふさ毛（細く短い差し毛をたくさん＝ふわっと見せる）
        for k in (-2, -1, 0, 1, 2):
            fx = cx + lean * 0.3 + k * ear_w * 0.075
            fy = base_y - ear_h * 0.02
            tuft = [(fx - ear_w * 0.03, fy),
                    (fx + ear_w * 0.03, fy),
                    (fx + k * ear_w * 0.008, fy - ear_h * 0.22)]
            d.polygon(tuft, fill=CAT_TUFT)

    target_h = int(CATEAR_BASE_H * width / CATEAR_BASE_W)
    img = img.resize((width, target_h), Image.LANCZOS)
    bgra = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    _CATEAR_CACHE[width] = bgra
    return bgra

def draw_sunglasses(frame, face_landmarks, sunglasses):
    """サングラスを顔に合成"""
    ih, iw = frame.shape[:2]
    lm = face_landmarks.landmark
    left_eye, right_eye = lm[33], lm[263]
    lx, ly = int(left_eye.x * iw), int(left_eye.y * ih)
    rx, ry = int(right_eye.x * iw), int(right_eye.y * ih)

    eye_w = math.hypot(rx - lx, ry - ly)
    width = int(eye_w * SUNG_WIDTH_FACTOR)
    if width < 10:
        return frame
    sh, sw = sunglasses.shape[:2]
    height = int(width * sh / sw)

    resized = cv2.resize(sunglasses, (width, height), interpolation=cv2.INTER_AREA)

    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    M = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(height * sin + width * cos)
    nH = int(height * cos + width * sin)
    M[0, 2] += nW / 2 - width / 2
    M[1, 2] += nH / 2 - height / 2
    rotated = cv2.warpAffine(resized, M, (nW, nH),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    center_x, center_y = (lx + rx) // 2, (ly + ry) // 2
    return overlay_rgba(frame, rotated, center_x - nW // 2, center_y - nH // 2)

def draw_cat_ears(frame, face_landmarks):
    """猫耳を顔に合成"""
    ih, iw = frame.shape[:2]
    lm = face_landmarks.landmark

    lf, rf = lm[234], lm[454]
    face_w = math.hypot((rf.x - lf.x) * iw, (rf.y - lf.y) * ih)
    width = int(face_w * 1.5)
    if width < 20:
        return frame
    ears = build_cat_ears(width)
    eh, ew = ears.shape[:2]

    le, re = lm[33], lm[263]
    lx, ly = le.x * iw, le.y * ih
    rx, ry = re.x * iw, re.y * ih
    angle = math.degrees(math.atan2(ry - ly, rx - lx))

    M = cv2.getRotationMatrix2D((ew / 2, eh / 2), -angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(eh * sin + ew * cos)
    nH = int(eh * cos + ew * sin)
    M[0, 2] += nW / 2 - ew / 2
    M[1, 2] += nH / 2 - eh / 2
    rotated = cv2.warpAffine(ears, M, (nW, nH),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    ang = math.radians(angle)
    up_x, up_y = math.sin(ang), -math.cos(ang)
    fhx, fhy = lm[10].x * iw, lm[10].y * ih
    shift = eh * 0.5
    cx = fhx + up_x * shift
    cy = fhy + up_y * shift
    return overlay_rgba(frame, rotated, int(cx - nW / 2), int(cy - nH / 2))

# ============================================================
# ポーズ判定
# ============================================================

def detect_two_hand_pose(hand_centers, face_top_y, iw, ih):
    """両手のポーズを判定"""
    if len(hand_centers) < 2:
        return None
    (x1, y1), (x2, y2) = hand_centers[0], hand_centers[1]

    if y1 < face_top_y and y2 < face_top_y:
        return "BANZAI"

    hands_close = math.hypot(x2 - x1, y2 - y1) < iw * 0.28
    below_face = y1 > face_top_y and y2 > face_top_y
    if hands_close and below_face:
        return "HEART"
    return None

def classify_body_pose(pose_landmarks):
    """全身ポーズを判定"""
    lm = pose_landmarks.landmark
    LS, RS = 11, 12
    LW, RW = 15, 16
    NOSE = 0

    sh_y = (lm[LS].y + lm[RS].y) / 2.0
    spread = abs(lm[LW].x - lm[RW].x)
    level = abs(lm[LW].y - sh_y) < 0.15 and abs(lm[RW].y - sh_y) < 0.15

    if spread > 0.55 and level:
        return "TPOSE"
    if lm[LW].y < lm[NOSE].y and lm[RW].y < lm[NOSE].y:
        return "BANZAI"
    return None

# ============================================================
# UI
# ============================================================

st.title("🎬 AI画像認識 ライブデモ")
st.caption("😎 顔を見つけると飾りがつき、✌️ ✋ ✊ 👍 🦊 のサインでエフェクトが出ます！")

with st.sidebar:
    st.header("⚙️ 設定")

    if st.button("← ホームに戻る", use_container_width=True):
        st.switch_page("app.py")

    st.markdown("---")

    with st.expander("🎬 カメラ設定", expanded=True):
        cam_width = st.slider("カメラ横幅", 480, 1280, CAM_WIDTH, step=160)
        cam_height = st.slider("カメラ縦幅", 270, 720, CAM_HEIGHT, step=90)
        infer_scale = st.select_slider("処理速度（小さいほど速い）",
                                       options=[0.25, 0.35, 0.5, 0.75, 1.0],
                                       value=INFER_SCALE)

    with st.expander("🎨 表示設定"):
        deco = st.selectbox("😎 顔の飾り",
                           ["サングラス😎", "猫耳🐱", "サングラス+猫耳", "なし"], index=0)
        show_sunglasses = "サングラス" in deco
        show_catears = "猫耳" in deco

    with st.expander("✨ エフェクト設定", expanded=True):
        show_fun = st.checkbox("🎉 おもしろポーズ（バンザイ/ハート）", value=True)
        use_pose = st.checkbox("🕺 全身ポーズ判定（少し重め）", value=True)
        show_landmarks = st.checkbox("🖐️ 手の骨格を表示", value=False)

    with st.expander("🔊 サウンド"):
        enable_sound = st.checkbox("効果音を鳴らす", value=False, disabled=not _AUDIO_OK)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button("▶️ スタート", use_container_width=True)
    with col2:
        stop_clicked = st.button("⏹️ 停止", use_container_width=True)

if "running" not in st.session_state:
    st.session_state.running = True
if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False

_vp_left, _vp_center, _vp_right = st.columns([1, 8, 1])
with _vp_center:
    video_placeholder = st.empty()
    info_placeholder = st.empty()

# ============================================================
# メインループ
# ============================================================

if st.session_state.running:
    sunglasses, fonts, emoji_font, emoji_ok = load_assets()
    if sunglasses is None:
        st.error(f"サングラス画像が読み込めませんでした: {SUNGLASSES_PATH}")
        st.stop()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        st.error("カメラが見つかりません。他のアプリがカメラを使っていないか確認してください。")
        st.stop()

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    gesture_buffers = [deque(maxlen=GESTURE_SMOOTH_N) for _ in range(MAX_HANDS)]
    last_sound = [None] * MAX_HANDS
    prev_time = time.time()
    fps = 0.0

    pose = mp_pose.Pose(model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) if use_pose else None

    with mp_face_mesh.FaceMesh(
            max_num_faces=MAX_FACES, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
         mp_hands.Hands(
            model_complexity=0, max_num_hands=MAX_HANDS,
            min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:

        while st.session_state.running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            ih, iw = frame.shape[:2]
            now = time.time()

            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb_full, None, fx=infer_scale, fy=infer_scale,
                               interpolation=cv2.INTER_AREA)
            small.flags.writeable = False
            face_results = face_mesh.process(small)
            hand_results = hands.process(small)
            pose_results = pose.process(small) if pose is not None else None

            text_jobs, emoji_jobs = [], []

            face_top_y = ih * 0.25
            if face_results.multi_face_landmarks:
                tops = [f.landmark[10].y * ih for f in face_results.multi_face_landmarks]
                face_top_y = min(tops)

            if (show_sunglasses or show_catears) and face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    if show_catears:
                        frame = draw_cat_ears(frame, face_landmarks)
                    if show_sunglasses:
                        frame = draw_sunglasses(frame, face_landmarks, sunglasses)

            hand_centers = []
            if hand_results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    if i >= MAX_HANDS:
                        break
                    if show_landmarks:
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                            mp_draw.DrawingSpec(color=(0, 128, 255), thickness=1))

                    lm9 = hand_landmarks.landmark[9]
                    center = (int(lm9.x * iw), int(lm9.y * ih))
                    hand_centers.append(center)

                    g = classify_gesture(hand_landmarks)
                    if g != "UNKNOWN":
                        gesture_buffers[i].append(g)

                    if gesture_buffers[i]:
                        cand, votes = Counter(gesture_buffers[i]).most_common(1)[0]
                        if votes >= GESTURE_MIN_VOTES:
                            add_gesture_effect(frame, cand, center, now,
                                               text_jobs, emoji_jobs)
                            if cand != last_sound[i]:
                                play_gesture_sound(cand, enable_sound)
                                last_sound[i] = cand
            else:
                for buf in gesture_buffers:
                    buf.clear()
                last_sound = [None] * MAX_HANDS

            if show_fun:
                pose2 = detect_two_hand_pose(hand_centers, face_top_y, iw, ih)
                if pose2 == "BANZAI":
                    draw_confetti(frame, now)
                    text_jobs.append(("バンザイ！", (iw // 2 - 130, 90), "xl", (60, 255, 255), 4))
                    if emoji_ok:
                        emoji_jobs.append(("🙌", (iw // 2 + 150, 80), 90))
                elif pose2 == "HEART":
                    mx = (hand_centers[0][0] + hand_centers[1][0]) // 2
                    my = (hand_centers[0][1] + hand_centers[1][1]) // 2
                    draw_heart(frame, (mx, my - 30), 70)
                    text_jobs.append(("だいすき♡", (mx - 110, my - 170), "l", (255, 120, 200), 3))

            if pose_results is not None and pose_results.pose_landmarks:
                bp = classify_body_pose(pose_results.pose_landmarks)
                if bp == "TPOSE":
                    text_jobs.append(("Tポーズ！", (iw // 2 - 130, 150), "xl", (60, 220, 90), 4))
                elif bp == "BANZAI":
                    draw_confetti(frame, now)
                    text_jobs.append(("バンザイ！", (iw // 2 - 130, 150), "xl", (60, 255, 255), 4))

            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (iw, 44), (40, 30, 30), -1)
            cv2.rectangle(overlay, (0, ih - 36), (iw, ih), (40, 30, 30), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            text_jobs.append(("AI画像認識 体験デモ", (14, 8), "s", (255, 255, 255), 2))
            text_jobs.append(("ピース・パー・グー・サムズアップ・きつねを見せてね！",
                              (14, ih - 30), "xs", (120, 230, 255), 2))

            cv2.putText(frame, f"FPS:{fps:4.1f}", (iw - 118, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            frame = render_text_layer(frame, text_jobs, fonts)
            frame = render_emoji_layer(frame, emoji_jobs, emoji_font, emoji_ok)

            video_placeholder.image(frame, channels="BGR",
                                    use_container_width=True, output_format="JPEG")

    if pose is not None:
        pose.close()
    cap.release()
    cv2.destroyAllWindows()
else:
    info_placeholder.info("左の「▶️ スタート」を押すとカメラが起動します。")
