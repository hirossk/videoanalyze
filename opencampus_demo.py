# -*- coding: utf-8 -*-
# ============================================================
#  🎓 AI画像認識 体験デモ（オープンキャンパス用）
# ------------------------------------------------------------
#  カメラを起動しっぱなしにして、来てくれた学生さんへ
#  「AIってこんなことができるんだ！」をアピールするデモです。
#
#   ・顔を見つけたら …… サングラス😎 / 猫耳🐱 を自動で装着
#   ・手を見つけたら …… ジェスチャーを判定してエフェクト！
#        ✌ ピース / ✋ パー / ✊ グー / 👍 サムズアップ / 🦊 きつね
#   ・両手や全身の「おもしろポーズ」も判定！
#        🙌 バンザイ（紙吹雪）/ 🫶 両手ハート / 🕺 Tポーズ
#
#  実行方法（ターミナルで）:
#      streamlit run opencampus_demo.py
#
#  ★ 動作が重い（FPSが低い）と感じたら、下の「設定」にある
#    CAM_WIDTH / CAM_HEIGHT / INFER_SCALE を小さくしてみてください。
# ============================================================

import time
import math
from collections import deque, Counter

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------
# 0. 設定（ここの数字を変えると見た目や判定の感度・速度が変わります）
# ------------------------------------------------------------
SUNGLASSES_PATH = "glasses/glasses1.png"     # かけるサングラスの画像
JP_FONT_PATH    = "fonts/NotoSansJP-Regular.ttf"  # 日本語フォント
EMOJI_FONT_PATH = "fonts/NotoColorEmoji.ttf"      # 絵文字フォント（使える環境だけ自動で使用）

# --- カメラ＆速度の設定（FPSを上げたいときの主役）---
CAM_WIDTH  = 960          # カメラの横サイズ（小さいほど速い：640などにすると更に軽い）
CAM_HEIGHT = 540          # カメラの縦サイズ
INFER_SCALE = 0.5         # AIに見せる画像の縮小率（0.5＝半分の解像度で推論＝約2倍速）

SUNG_WIDTH_FACTOR = 2.2   # サングラスの幅を「両目の間」の何倍にするか
MAX_FACES = 2             # 同時に処理する顔の数
MAX_HANDS = 2             # 同時に処理する手の数

# ジェスチャー判定のしきい値（指が伸びている／親指が開いているの感度）
FINGER_EXT_MARGIN = 0.12  # 指が「伸びている」とみなす余裕（大きいほど判定が厳しい＝甘くない）
THUMB_OPEN_RATIO  = 0.50  # 親指が「開いている」とみなす距離の比率
GESTURE_SMOOTH_N  = 5      # 直近Nフレームの多数決でブレを抑える
GESTURE_MIN_VOTES = 3      # N回中、最低この回数そろって初めて確定（誤検出を減らす）

# 日本語ラベル（画面に出す言葉）と、テーマカラー（B, G, R）
GESTURE_INFO = {
    "PEACE": {"label": "ピース！",       "color": (255, 120,  60)},   # 青系
    "PA":    {"label": "パー！",         "color": ( 60, 200, 255)},   # 黄系
    "GU":    {"label": "グー！",         "color": ( 80,  80, 255)},   # 赤系
    "GOOD":  {"label": "いいね！ GOOD",  "color": ( 60, 220,  90)},   # 緑系
    "FOX":   {"label": "きつね！",       "color": ( 30, 150, 255)},   # オレンジ系
}
EMOJI_MAP = {"PEACE": "✌", "PA": "✋", "GU": "✊", "GOOD": "👍", "FOX": "🦊"}


# ------------------------------------------------------------
# 効果音（グー・パー・チョキ）。音が出ない環境でも止まらないように保護する
# ------------------------------------------------------------
try:
    import sounddevice as sd
    _AUDIO_OK = True
except Exception:
    _AUDIO_OK = False

SAMPLE_RATE = 44100


def _make_tone(freqs, dur=0.18, volume=0.3):
    """指定した周波数（複数なら和音）の短い音を作る。プツッと鳴らないよう端をフェード。"""
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), False)
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    wave = wave / (np.max(np.abs(wave)) + 1e-9)
    fade = int(SAMPLE_RATE * 0.012)
    env = np.ones_like(wave)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (wave * env * volume).astype(np.float32)


# じゃんけんの3種類に、それぞれ違う音色を割り当てる
GESTURE_SOUNDS = {
    "GU":    _make_tone([196, 247]),            # グー：低めの和音
    "PA":    _make_tone([523, 659, 784]),       # パー：明るい和音（ドミソ）
    "PEACE": _make_tone([440, 554, 659]),       # チョキ：少し高めの和音
} if _AUDIO_OK else {}


def play_gesture_sound(gesture, enabled):
    """ジェスチャーに対応した効果音を鳴らす（鳴らせないときは何もしない）"""
    if enabled and _AUDIO_OK and gesture in GESTURE_SOUNDS:
        try:
            sd.play(GESTURE_SOUNDS[gesture], SAMPLE_RATE)
        except Exception:
            pass


# ============================================================
# 1. 画像・フォントの読み込み（一度だけ実行して使いまわす）
# ============================================================
@st.cache_resource
def load_assets():
    """サングラス画像と日本語/絵文字フォントを読み込む"""
    sunglasses = cv2.imread(SUNGLASSES_PATH, cv2.IMREAD_UNCHANGED)
    # アルファ（透明）チャンネルが無ければ追加して4チャンネルにそろえる
    if sunglasses is not None and sunglasses.shape[2] == 3:
        h, w = sunglasses.shape[:2]
        alpha = np.full((h, w, 1), 255, dtype=sunglasses.dtype)
        sunglasses = np.concatenate([sunglasses, alpha], axis=2)

    fonts = {}
    for name, size in [("xs", 20), ("s", 28), ("m", 40), ("l", 64), ("xl", 96)]:
        fonts[name] = ImageFont.truetype(JP_FONT_PATH, size)

    # 絵文字フォントは環境によって描画できないことがあるので、実際に試す
    emoji_font = None
    emoji_ok = False
    try:
        ef = ImageFont.truetype(EMOJI_FONT_PATH, 109)
        test = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(test).text((10, 10), "👍", font=ef, embedded_color=True)
        if np.array(test)[:, :, 3].any():   # 何か描けていれば使える
            emoji_font = ef
            emoji_ok = True
    except Exception:
        emoji_ok = False

    return sunglasses, fonts, emoji_font, emoji_ok


# ============================================================
# 2. 便利な道具（描画ヘルパー）
# ============================================================
def overlay_rgba(background, overlay, x, y):
    """透明付き画像(overlay, 4ch)を background(3ch) の (x, y) に貼り付ける。
    画面からはみ出す部分は自動で切り取る。"""
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # 貼り付け先の範囲を画面内にクリップ
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + ow), min(bh, y + oh)
    if x1 >= x2 or y1 >= y2:
        return background  # 完全に画面の外

    # overlay 側の対応する範囲
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    roi = background[y1:y2, x1:x2].astype(np.float32)
    ov = overlay[oy1:oy2, ox1:ox2].astype(np.float32)
    alpha = (ov[:, :, 3:4]) / 255.0
    blended = alpha * ov[:, :, :3] + (1 - alpha) * roi
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


# --- 絵文字パッチのキャッシュ（同じ絵文字を毎回作り直さない＝速い）---
_EMOJI_CACHE = {}


def get_emoji_bgra(emoji_font, emoji_char, size):
    """絵文字を size×size のBGRA画像にして返す（作ったものは覚えておく）"""
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
    """ためておいた日本語テキストを「1回だけ」のPIL変換でまとめて描く。
    （フレームごとに何度もPIL変換すると重いので、最後に一括で処理）"""
    if not text_jobs:
        return frame
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, pos, fkey, rgb, stroke in text_jobs:
        draw.text(pos, text, font=fonts[fkey], fill=rgb,
                  stroke_width=stroke, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def render_emoji_layer(frame, emoji_jobs, emoji_font, emoji_ok):
    """ためておいた絵文字をまとめて貼り付ける"""
    if not emoji_ok:
        return frame
    for char, (x, y), size in emoji_jobs:
        patch = get_emoji_bgra(emoji_font, char, size)
        frame = overlay_rgba(frame, patch, x, y)
    return frame


# ============================================================
# 3. ジェスチャー判定ロジック（精度を高めたバージョン）
# ============================================================
def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _fingers_extended(lm, hand_size):
    """親指以外の4本について「伸びているか」を判定して [人,中,薬,小] で返す。
    手首からの距離で測るので手の向きに強い。さらにマージンを付けて誤検出を減らす。"""
    wrist = lm[0]
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    result = []
    for tip, pip in zip(finger_tips, finger_pips):
        # 指先がPIP関節より「手の大きさ × マージン」ぶん手首から遠ければ伸びている
        diff = (_dist(lm[tip], wrist) - _dist(lm[pip], wrist)) / hand_size
        result.append(diff > FINGER_EXT_MARGIN)
    return result


def classify_gesture(hand_landmarks):
    """手のランドマークから ピース/パー/グー/サムズアップ/きつね を判定する。"""
    lm = hand_landmarks.landmark
    wrist = lm[0]
    hand_size = _dist(wrist, lm[9]) + 1e-6     # 手の大きさ（手首〜中指の付け根）

    idx, mid, ring, pinky = _fingers_extended(lm, hand_size)
    n_ext = sum([idx, mid, ring, pinky])

    # 親指：付け根(2)から離れて開いているか / 付け根よりはっきり上を向いているか
    thumb_open = (_dist(lm[4], lm[2]) / hand_size) > THUMB_OPEN_RATIO
    thumb_up = (lm[4].y < lm[2].y - 0.04) and (lm[4].y < wrist.y)

    # --- 判定（特徴的なものから先に調べる）---
    if idx and mid and ring and pinky and thumb_open:
        return "PA"                                   # パー（全部開く）
    if idx and pinky and (not mid) and (not ring):
        return "FOX"                                  # きつね（人差し指＋小指）
    if idx and mid and (not ring) and (not pinky):
        return "PEACE"                                # ピース（人差し指＋中指）
    if n_ext == 0 and thumb_open and thumb_up:
        return "GOOD"                                 # サムズアップ（親指だけ立てて上向き）
    if n_ext == 0:
        return "GU"                                   # グー（にぎりこぶし）
    return "UNKNOWN"


# ============================================================
# 4. エフェクト描画（OpenCVだけで直接描く部分）
# ============================================================
def add_gesture_effect(frame, gesture, center, t, text_jobs, emoji_jobs):
    """ジェスチャーに合わせて手の近くに派手なエフェクトを描く。
    リングや星はその場でOpenCV描画し、日本語と絵文字は後でまとめて描くために予約する。"""
    info = GESTURE_INFO[gesture]
    color = info["color"]
    cx, cy = center

    # 広がるリング（time でアニメーション）
    phase = (t * 1.5) % 1.0
    for k in range(3):
        r = int(40 + ((phase + k / 3.0) % 1.0) * 90)
        cv2.circle(frame, (cx, cy), r, color, 3, cv2.LINE_AA)

    # きらめき（手の周りに回る星）
    for k in range(6):
        ang = t * 2 + k * (2 * math.pi / 6)
        sx = int(cx + math.cos(ang) * 75)
        sy = int(cy + math.sin(ang) * 75)
        cv2.drawMarker(frame, (sx, sy), color, cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)

    # 日本語ラベルと絵文字は「あとでまとめて描く」ために予約しておく
    rgb = (color[2], color[1], color[0])              # BGR -> RGB
    text_jobs.append((info["label"], (cx - 90, cy - 150), "l", rgb, 3))
    if gesture in EMOJI_MAP:
        emoji_jobs.append((EMOJI_MAP[gesture], (cx + 95, cy - 150), 70))


def draw_confetti(frame, t):
    """バンザイのときに降らせる紙吹雪（軽いのでOpenCVで直接描く）"""
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
    """指定位置にハートマークを描く（2つの円＋三角形）"""
    cx, cy = center
    r = max(6, size // 4)
    cv2.circle(frame, (cx - r, cy - r // 2), r, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + r, cy - r // 2), r, color, -1, cv2.LINE_AA)
    pts = np.array([[cx - 2 * r, cy - r // 4],
                    [cx + 2 * r, cy - r // 4],
                    [cx, cy + 2 * r]], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts, color, cv2.LINE_AA)


# ============================================================
# 5. 顔の飾り（サングラス・猫耳）
# ============================================================
# --- かわいい猫耳を作る（曲線で描いてキャッシュ）---
_CATEAR_CACHE = {}
CATEAR_BASE_W = 400          # 元絵の縦横（実際は顔の幅に合わせて縮小して使う）
CATEAR_BASE_H = 300

# 猫耳の色（お好みで変えてOK）：毛＝こげ茶グレー / 内耳＝ピンク
CAT_FUR     = (70, 60, 78, 255)
CAT_FUR_LINE = (40, 34, 46, 255)
CAT_PINK    = (235, 150, 195, 255)
CAT_PINK_LT = (245, 200, 220, 255)


def _quad_bezier(p0, c, p1, n=16):
    """2次ベジェ曲線をn個の点でサンプリング（耳の丸みを作る）"""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t ** 2 * p1[1]
        pts.append((x, y))
    return pts


def _ear_polygon(cx, base_y, w, h, tip_dx=0.0):
    """丸い先っぽの猫耳の輪郭を点の列で返す。tip_dxで先っぽを左右に倒せる。"""
    tx = cx + tip_dx                              # 先っぽのx（外側に倒すと猫らしい）
    bl = (cx - w / 2, base_y)
    br = (cx + w / 2, base_y)
    tl = (tx - w * 0.12, base_y - h)              # 先っぽは点ではなく少し幅をもたせて丸く
    tr = (tx + w * 0.12, base_y - h)
    cL = (cx - w * 0.52, base_y - h * 0.50)       # 左側のふくらみ
    cR = (cx + w * 0.52, base_y - h * 0.50)       # 右側のふくらみ
    cap = (tx, base_y - h * 1.12)                 # 上のドーム（丸い先端）
    return (_quad_bezier(bl, cL, tl)
            + _quad_bezier(tl, cap, tr, n=8)
            + _quad_bezier(tr, cR, br))


def build_cat_ears(width):
    """指定した幅のかわいい猫耳（両耳）をBGRA画像で作る。一度作ったら覚えておく。"""
    width = max(40, int(round(width / 8) * 8))     # キャッシュが増えすぎないよう8刻みに
    if width in _CATEAR_CACHE:
        return _CATEAR_CACHE[width]

    SS = 4                                          # 4倍で描いて縮小＝なめらかな輪郭に
    W, H = CATEAR_BASE_W * SS, CATEAR_BASE_H * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    ear_w, ear_h = W * 0.40, H * 0.72
    base_y = H * 0.95
    for side in (-1, 1):                            # 左耳 / 右耳
        cx = W / 2 + side * W * 0.205
        lean = side * ear_w * 0.20                  # 先っぽを少し外側へ倒す＝猫らしい
        outer = _ear_polygon(cx, base_y, ear_w, ear_h, lean)
        d.polygon(outer, fill=CAT_FUR)
        d.line(outer + [outer[0]], fill=CAT_FUR_LINE, width=int(SS * 3), joint="curve")
        # 内耳のピンク（少し上にずらして小さく）
        inner = _ear_polygon(cx, base_y - ear_h * 0.05, ear_w * 0.54, ear_h * 0.66, lean)
        d.polygon(inner, fill=CAT_PINK)
        inner2 = _ear_polygon(cx, base_y - ear_h * 0.08, ear_w * 0.32, ear_h * 0.48, lean)
        d.polygon(inner2, fill=CAT_PINK_LT)

    target_h = int(CATEAR_BASE_H * width / CATEAR_BASE_W)
    img = img.resize((width, target_h), Image.LANCZOS)
    bgra = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    _CATEAR_CACHE[width] = bgra
    return bgra


def draw_sunglasses(frame, face_landmarks, sunglasses):
    """顔のランドマークから両目の位置を求め、傾きに合わせてサングラスを合成する"""
    ih, iw = frame.shape[:2]
    lm = face_landmarks.landmark
    left_eye, right_eye = lm[33], lm[263]   # 左右の目尻
    lx, ly = int(left_eye.x * iw), int(left_eye.y * ih)
    rx, ry = int(right_eye.x * iw), int(right_eye.y * ih)

    eye_w = math.hypot(rx - lx, ry - ly)
    width = int(eye_w * SUNG_WIDTH_FACTOR)
    if width < 10:
        return frame
    sh, sw = sunglasses.shape[:2]
    height = int(width * sh / sw)

    resized = cv2.resize(sunglasses, (width, height), interpolation=cv2.INTER_AREA)

    # 顔の傾き（両目を結ぶ線の角度）に合わせてサングラスも回転させる
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
    """かわいい猫耳を、顔の幅と傾きに合わせて頭の上に合成する"""
    ih, iw = frame.shape[:2]
    lm = face_landmarks.landmark

    # 顔の幅（左右の輪郭）に合わせて猫耳の大きさを決める
    lf, rf = lm[234], lm[454]
    face_w = math.hypot((rf.x - lf.x) * iw, (rf.y - lf.y) * ih)
    width = int(face_w * 1.5)
    if width < 20:
        return frame
    ears = build_cat_ears(width)
    eh, ew = ears.shape[:2]

    # 顔の傾き（両目を結ぶ線の角度）に合わせて猫耳も回す
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

    # おでこ(10)の少し上を中心にして、頭の「上方向」へずらして乗せる
    ang = math.radians(angle)
    up_x, up_y = math.sin(ang), -math.cos(ang)
    fhx, fhy = lm[10].x * iw, lm[10].y * ih
    shift = eh * 0.5
    cx = fhx + up_x * shift
    cy = fhy + up_y * shift
    return overlay_rgba(frame, rotated, int(cx - nW / 2), int(cy - nH / 2))


# ============================================================
# 6. おもしろポーズ判定（両手・全身）
# ============================================================
def detect_two_hand_pose(hand_centers, face_top_y, iw, ih):
    """両手の位置関係から「バンザイ」「両手ハート」を判定する。
    hand_centers: [(x, y), ...] 画面ピクセル座標 / face_top_y: 顔の上端の高さ"""
    if len(hand_centers) < 2:
        return None
    (x1, y1), (x2, y2) = hand_centers[0], hand_centers[1]

    # バンザイ：両手とも顔の上端より高い位置にある
    if y1 < face_top_y and y2 < face_top_y:
        return "BANZAI"

    # 両手ハート：両手が近づいていて、顔より下（胸のあたり）にある
    hands_close = math.hypot(x2 - x1, y2 - y1) < iw * 0.28
    below_face = y1 > face_top_y and y2 > face_top_y
    if hands_close and below_face:
        return "HEART"
    return None


def classify_body_pose(pose_landmarks):
    """全身の関節から「Tポーズ」「バンザイ」を判定する（MediaPipe Pose使用）"""
    lm = pose_landmarks.landmark
    LS, RS = 11, 12      # 左右の肩
    LW, RW = 15, 16      # 左右の手首
    NOSE = 0

    sh_y = (lm[LS].y + lm[RS].y) / 2.0
    spread = abs(lm[LW].x - lm[RW].x)                       # 両手首の左右の開き
    level = abs(lm[LW].y - sh_y) < 0.15 and abs(lm[RW].y - sh_y) < 0.15

    # Tポーズ：両腕を横にまっすぐ広げる（手首が肩の高さ＆大きく開く）
    if spread > 0.55 and level:
        return "TPOSE"
    # バンザイ：両手首が顔（鼻）より上
    if lm[LW].y < lm[NOSE].y and lm[RW].y < lm[NOSE].y:
        return "BANZAI"
    return None


# ============================================================
# 7. 案内バナー
# ============================================================
def draw_banner(frame, fps, text_jobs):
    """画面の上下に、学生さんへの案内バナーを細めに描く（文字は後でまとめて描く）"""
    ih, iw = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (iw, 44), (40, 30, 30), -1)
    cv2.rectangle(overlay, (0, ih - 36), (iw, ih), (40, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    text_jobs.append(("AI画像認識 体験デモ", (14, 8), "s", (255, 255, 255), 2))
    text_jobs.append(("ピース・パー・グー・サムズアップ・きつねを見せてね！",
                      (14, ih - 30), "xs", (120, 230, 255), 2))

    cv2.putText(frame, f"FPS:{fps:4.1f}", (iw - 118, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ============================================================
# 8. Streamlit 画面（UI）
# ============================================================
st.set_page_config(page_title="AI体験デモ｜オープンキャンパス", page_icon="🎓", layout="wide")
st.title("🎓 AI画像認識 体験デモ")
st.caption("😎 顔を見つけると飾り／✌ ✋ ✊ 👍 🦊 の手のサインや 🙌 🫶 のポーズでエフェクトが出ます！")

with st.sidebar:
    st.header("コントロール")
    start_clicked = st.button("▶️ スタート", use_container_width=True)
    stop_clicked = st.button("⏹️ 停止", use_container_width=True)
    st.markdown("---")
    deco = st.selectbox("😎 顔の飾り",
                        ["サングラス😎", "猫耳🐱", "サングラス+猫耳", "なし"], index=0)
    show_sunglasses = "サングラス" in deco
    show_catears = "猫耳" in deco
    show_fun = st.checkbox("🎉 おもしろポーズ判定（バンザイ/ハート）", value=True)
    use_pose = st.checkbox("🕺 全身ポーズも判定（Tポーズ等・少し重め）", value=False)
    enable_sound = st.checkbox("🔊 効果音（グー/パー/チョキ）", value=_AUDIO_OK,
                               disabled=not _AUDIO_OK)
    show_landmarks = st.checkbox("🖐️ 手の骨格を表示", value=False)
    st.markdown("---")
    st.markdown(
        "**遊び方**\n\n"
        "1. カメラに顔を向けると飾りがつくよ😎🐱\n"
        "2. 手のサインを見せてね！\n"
        "    - ✌ ピース / ✋ パー / ✊ グー\n"
        "    - 👍 サムズアップ / 🦊 きつね\n"
        "3. ポーズにも挑戦！\n"
        "    - 🙌 両手を上げてバンザイ\n"
        "    - 🫶 両手を近づけてハート\n"
        "    - 🕺 Tポーズ（全身ポーズON時）"
    )

# オープンキャンパス用：初回から自動でカメラON（常にアピール）
if "running" not in st.session_state:
    st.session_state.running = True
if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False

# カメラ映像は画面いっぱいだと大きすぎるので、中央 80% 幅に収める
_vp_left, _vp_center, _vp_right = st.columns([1, 8, 1])
with _vp_center:
    video_placeholder = st.empty()
    info_placeholder = st.empty()


# ============================================================
# 9. メインループ（カメラ映像をひたすら処理して表示）
# ============================================================
if st.session_state.running:
    sunglasses, fonts, emoji_font, emoji_ok = load_assets()
    if sunglasses is None:
        st.error(f"サングラス画像が読み込めませんでした: {SUNGLASSES_PATH}")
        st.stop()

    # Windowsでは DSHOW を指定するとカメラが速く開く
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # 遅延を減らす
    if not cap.isOpened():
        st.error("カメラが見つかりません。他のアプリがカメラを使っていないか確認してください。")
        st.stop()

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    # 手ごとにジェスチャーのブレを抑えるための履歴
    gesture_buffers = [deque(maxlen=GESTURE_SMOOTH_N) for _ in range(MAX_HANDS)]
    last_sound = [None] * MAX_HANDS          # 手ごとに「最後に鳴らしたサイン」を覚えておく
    prev_time = time.time()
    fps = 0.0

    # 全身ポーズはONのときだけ作る（重いので）
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

            frame = cv2.flip(frame, 1)               # 鏡のように左右反転
            ih, iw = frame.shape[:2]
            now = time.time()

            # --- AIに見せる画像は小さくして高速化（座標は正規化なのでズレない）---
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb_full, None, fx=INFER_SCALE, fy=INFER_SCALE,
                               interpolation=cv2.INTER_AREA)
            small.flags.writeable = False
            face_results = face_mesh.process(small)
            hand_results = hands.process(small)
            pose_results = pose.process(small) if pose is not None else None

            # このフレームで描く「日本語テキスト」「絵文字」をためる入れもの
            text_jobs, emoji_jobs = [], []

            # 顔の上端（バンザイ判定などに使う）
            face_top_y = ih * 0.25
            if face_results.multi_face_landmarks:
                tops = [f.landmark[10].y * ih for f in face_results.multi_face_landmarks]
                face_top_y = min(tops)

            # --- 顔：飾りを合成（サングラス／猫耳）---
            if (show_sunglasses or show_catears) and face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    if show_catears:
                        frame = draw_cat_ears(frame, face_landmarks)
                    if show_sunglasses:
                        frame = draw_sunglasses(frame, face_landmarks, sunglasses)

            # --- 手：ジェスチャー判定 → エフェクト ---
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

                    # 直近フレームの多数決＋最低票数で「確実なときだけ」表示する
                    if gesture_buffers[i]:
                        cand, votes = Counter(gesture_buffers[i]).most_common(1)[0]
                        if votes >= GESTURE_MIN_VOTES:
                            add_gesture_effect(frame, cand, center, now,
                                               text_jobs, emoji_jobs)
                            # サインが切り替わった瞬間だけ効果音を鳴らす
                            if cand != last_sound[i]:
                                play_gesture_sound(cand, enable_sound)
                                last_sound[i] = cand
            else:
                # 手が消えたら履歴をリセット（古いサインが残らないように）
                for buf in gesture_buffers:
                    buf.clear()
                last_sound = [None] * MAX_HANDS

            # --- おもしろポーズ（両手）---
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

            # --- 全身ポーズ（Tポーズ等）---
            if pose_results is not None and pose_results.pose_landmarks:
                bp = classify_body_pose(pose_results.pose_landmarks)
                if bp == "TPOSE":
                    text_jobs.append(("Tポーズ！", (iw // 2 - 130, 150), "xl", (60, 220, 90), 4))
                elif bp == "BANZAI":
                    draw_confetti(frame, now)
                    text_jobs.append(("バンザイ！", (iw // 2 - 130, 150), "xl", (60, 255, 255), 4))

            # --- FPS 計算 & 案内バナー ---
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            frame = draw_banner(frame, fps, text_jobs)

            # --- ためておいた日本語＆絵文字を「1回だけ」まとめて描画（高速化）---
            frame = render_text_layer(frame, text_jobs, fonts)
            frame = render_emoji_layer(frame, emoji_jobs, emoji_font, emoji_ok)

            # JPEGで送ると転送が軽い
            video_placeholder.image(frame, channels="BGR",
                                    use_container_width=True, output_format="JPEG")

    if pose is not None:
        pose.close()
    cap.release()
    cv2.destroyAllWindows()
else:
    info_placeholder.info("左の「▶️ スタート」を押すとカメラが起動します。")
