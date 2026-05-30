# -*- coding: utf-8 -*-
# ============================================================
#  🎓 AI画像認識 体験デモ（オープンキャンパス用）
# ------------------------------------------------------------
#  カメラを起動しっぱなしにして、来てくれた学生さんへ
#  「AIってこんなことができるんだ！」をアピールするデモです。
#
#   ・顔を見つけたら …… サングラスを自動で装着 😎
#   ・手を見つけたら …… ジェスチャーを判定してエフェクト！
#        ✌ ピース / ✋ パー / ✊ グー / 👍 サムズアップ
#
#  実行方法（ターミナルで）:
#      streamlit run opencampus_demo.py
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
# 0. 設定（ここの数字を変えると見た目や判定の感度が変わります）
# ------------------------------------------------------------
SUNGLASSES_PATH = "glasses/glasses1.png"     # かけるサングラスの画像
JP_FONT_PATH    = "fonts/NotoSansJP-Regular.ttf"  # 日本語フォント
EMOJI_FONT_PATH = "fonts/NotoColorEmoji.ttf"      # 絵文字フォント（使える環境だけ自動で使用）

SUNG_WIDTH_FACTOR = 2.2   # サングラスの幅を「両目の間」の何倍にするか
MAX_FACES = 2             # 同時に処理する顔の数
MAX_HANDS = 2             # 同時に処理する手の数

# ジェスチャー判定のしきい値（指が伸びている／親指が開いているの感度）
THUMB_OPEN_RATIO = 0.55   # 親指が「開いている」とみなす距離の比率
GESTURE_SMOOTH_N = 5      # 直近Nフレームの多数決でブレを抑える

# 日本語ラベル（画面に出す言葉）と、テーマカラー（B, G, R）
GESTURE_INFO = {
    "PEACE": {"label": "ピース！",       "color": (255, 120,  60)},   # 青系
    "PA":    {"label": "パー！",         "color": ( 60, 200, 255)},   # 黄系
    "GU":    {"label": "グー！",         "color": ( 80,  80, 255)},   # 赤系
    "GOOD":  {"label": "いいね！ GOOD",  "color": ( 60, 220,  90)},   # 緑系
}


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
    for name, size in [("s", 28), ("m", 40), ("l", 64), ("xl", 96)]:
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


def draw_japanese(img_pil_draw, text, pos, font, fill, stroke=3, stroke_fill=(0, 0, 0)):
    """PIL で日本語を縁取り付きで描く（映像の上でも読みやすいように）"""
    img_pil_draw.text(pos, text, font=font, fill=fill,
                      stroke_width=stroke, stroke_fill=stroke_fill)


def make_emoji_patch(emoji_font, emoji_char, size):
    """絵文字を指定サイズのRGBA画像にして返す（使える環境のときだけ呼ぶ）"""
    patch = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((2, 2), emoji_char, font=emoji_font, embedded_color=True)
    bbox = patch.getbbox()
    if bbox:
        patch = patch.crop(bbox)
    return patch.resize((size, size), Image.LANCZOS)


# ============================================================
# 3. ジェスチャー判定ロジック
# ============================================================
def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def classify_gesture(hand_landmarks):
    """手のランドマークから ピース/パー/グー/サムズアップ を判定する。
    回転に強いように『手首からの距離』で指の伸び具合を測る。"""
    lm = hand_landmarks.landmark
    wrist = lm[0]

    # 手の大きさ（手首〜中指の付け根）を基準にして、しきい値をスケールさせる
    hand_size = _dist(wrist, lm[9]) + 1e-6

    # 親指以外の4本：指先が第二関節より手首から遠ければ「伸びている」
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    extended = []
    for tip, pip in zip(finger_tips, finger_pips):
        extended.append(_dist(lm[tip], wrist) > _dist(lm[pip], wrist))
    n_ext = sum(extended)

    # 親指：付け根(2)から離れて開いているか / 親指の付け根より上を向いているか
    thumb_open = (_dist(lm[4], lm[2]) / hand_size) > THUMB_OPEN_RATIO
    thumb_up = lm[4].y < lm[2].y - 0.03   # 親指の先が、付け根よりはっきり上

    # --- 判定（順番が大事：特徴的なものから先に調べる）---
    if n_ext == 4 and thumb_open:
        return "PA"                                   # パー（全部開く）
    if extended == [True, True, False, False]:
        return "PEACE"                                # ピース（人差し指＋中指）
    if n_ext == 0 and thumb_open and thumb_up:
        return "GOOD"                                 # サムズアップ（親指だけ立てて上向き）
    if n_ext == 0:
        return "GU"                                   # グー（にぎりこぶし）
    return "UNKNOWN"


# ============================================================
# 4. エフェクト描画
# ============================================================
def draw_gesture_effect(frame, gesture, center, fonts, emoji_font, emoji_ok, t):
    """ジェスチャーに合わせて、手の近くに派手なエフェクトを描く。
    frame: BGR画像 / center: 手の中心(x, y) / t: 時間（アニメ用）"""
    info = GESTURE_INFO[gesture]
    color = info["color"]
    cx, cy = center

    # --- 広がるリング（time でアニメーション）---
    phase = (t * 1.5) % 1.0
    for k in range(3):
        r = int(40 + ((phase + k / 3.0) % 1.0) * 90)
        cv2.circle(frame, (cx, cy), r, color, 3, cv2.LINE_AA)

    # --- きらめき（手の周りにランダムっぽい星）---
    for k in range(6):
        ang = t * 2 + k * (2 * math.pi / 6)
        sx = int(cx + math.cos(ang) * 75)
        sy = int(cy + math.sin(ang) * 75)
        cv2.drawMarker(frame, (sx, sy), color, cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)

    # --- ラベル（日本語）と絵文字を PIL で描く ---
    emoji_map = {"PEACE": "✌", "PA": "✋", "GU": "✊", "GOOD": "👍"}
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    label = info["label"]
    # BGR -> RGB（PILはRGB）
    rgb = (color[2], color[1], color[0])
    tx, ty = cx - 90, cy - 150
    draw_japanese(draw, label, (tx, ty), fonts["l"], rgb)
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    if emoji_ok:
        patch = make_emoji_patch(emoji_font, emoji_map[gesture], 70)
        patch_bgr = cv2.cvtColor(np.array(patch), cv2.COLOR_RGBA2BGRA)
        frame = overlay_rgba(frame, patch_bgr, cx + 95, cy - 150)

    return frame


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


def draw_banner(frame, fonts, fps):
    """画面の上下に、学生さんへの案内バナーを描く"""
    ih, iw = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (iw, 70), (40, 30, 30), -1)
    cv2.rectangle(overlay, (0, ih - 56), (iw, ih), (40, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw_japanese(draw, "AI画像認識 体験デモ", (16, 8), fonts["m"], (255, 255, 255), stroke=2)
    draw_japanese(draw, "ピース・パー・グー・サムズアップを見せてね！",
                  (16, ih - 50), fonts["s"], (120, 230, 255), stroke=2)
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.putText(frame, f"FPS:{fps:4.1f}", (iw - 140, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ============================================================
# 5. Streamlit 画面（UI）
# ============================================================
st.set_page_config(page_title="AI体験デモ｜オープンキャンパス", page_icon="🎓", layout="wide")
st.title("🎓 AI画像認識 体験デモ")
st.caption("😎 顔を見つけるとサングラス／✌ ✋ ✊ 👍 の手のサインを見せるとエフェクトが出ます！")

with st.sidebar:
    st.header("コントロール")
    start_clicked = st.button("▶️ スタート", use_container_width=True)
    stop_clicked = st.button("⏹️ 停止", use_container_width=True)
    st.markdown("---")
    show_sunglasses = st.checkbox("😎 サングラスをかける", value=True)
    show_landmarks = st.checkbox("🖐️ 手の骨格を表示", value=False)
    st.markdown("---")
    st.markdown(
        "**遊び方**\n\n"
        "1. カメラに顔を向けるとサングラス😎\n"
        "2. 手のサインを見せてね！\n"
        "    - ✌ ピース\n"
        "    - ✋ パー\n"
        "    - ✊ グー\n"
        "    - 👍 サムズアップ"
    )

# オープンキャンパス用：初回から自動でカメラON（常にアピール）
if "running" not in st.session_state:
    st.session_state.running = True
if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False

video_placeholder = st.empty()
info_placeholder = st.empty()


# ============================================================
# 6. メインループ（カメラ映像をひたすら処理して表示）
# ============================================================
if st.session_state.running:
    sunglasses, fonts, emoji_font, emoji_ok = load_assets()
    if sunglasses is None:
        st.error(f"サングラス画像が読み込めませんでした: {SUNGLASSES_PATH}")
        st.stop()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("カメラが見つかりません。他のアプリがカメラを使っていないか確認してください。")
        st.stop()

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # 手ごとにジェスチャーのブレを抑えるための履歴
    gesture_buffers = [deque(maxlen=GESTURE_SMOOTH_N) for _ in range(MAX_HANDS)]
    prev_time = time.time()
    fps = 0.0

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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False              # 高速化のおまじない
            face_results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)
            ih, iw = frame.shape[:2]
            now = time.time()

            # --- 顔：サングラスを合成 ---
            if show_sunglasses and face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    frame = draw_sunglasses(frame, face_landmarks, sunglasses)

            # --- 手：ジェスチャー判定 → エフェクト ---
            if hand_results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    if i >= MAX_HANDS:
                        break
                    if show_landmarks:
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                            mp_draw.DrawingSpec(color=(0, 128, 255), thickness=1))

                    g = classify_gesture(hand_landmarks)
                    if g != "UNKNOWN":
                        gesture_buffers[i].append(g)

                    # 直近フレームの多数決で安定したジェスチャーを採用
                    if gesture_buffers[i]:
                        stable = Counter(gesture_buffers[i]).most_common(1)[0][0]
                        # 手の中心（中指の付け根あたり）を計算
                        lm9 = hand_landmarks.landmark[9]
                        center = (int(lm9.x * iw), int(lm9.y * ih))
                        frame = draw_gesture_effect(
                            frame, stable, center, fonts, emoji_font, emoji_ok, now)
            else:
                # 手が消えたら履歴をリセット（古いサインが残らないように）
                for buf in gesture_buffers:
                    buf.clear()

            # --- FPS 計算 & 案内バナー ---
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            frame = draw_banner(frame, fonts, fps)

            video_placeholder.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()
else:
    info_placeholder.info("左の「▶️ スタート」を押すとカメラが起動します。")
