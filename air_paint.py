
# -*- coding: utf-8 -*-
"""
Streamlit + MediaPipe で作る エアペイント（空中お絵描き）
---------------------------------------------------------
インストール:
    pip install streamlit opencv-python mediapipe

実行:
    streamlit run air_paint.py

操作:
  - 左サイドバーで「🎨 開始」を押すとカメラ起動
  - 親指(4) と 人差し指(8) の先を「つまむ（ピンチ）」と描画開始、
    離すと描画停止（移動のみ）
  - 色, 太さ, 透明度, クリア/アンドゥ/保存 はサイドバーで操作
  - 画面上に現在の手の形と状態をオーバーレイ表示

学びポイント:
  - ランドマークの正規化座標 → ピンチ距離で状態判定
  - 2点間を結ぶ線描画(cv2.line)で連続ストロークを実現
  - ストローク履歴を保持→Undoのしくみ
"""
import time
import os
from collections import deque

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# ------------------------------
# ユーティリティ
# ------------------------------
def hex_to_bgr(hex_color: str):
    """#RRGGBB -> (B, G, R)"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def draw_text_with_bg(img, text, pos=(20, 40), scale=0.8, color=(255,255,255), bg=(0,0,0), thickness=2):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x-8, y-h-10), (x+w+8, y+8), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def ensure_layer(layer, shape):
    """描画レイヤー(黒)を初期化/サイズ調整"""
    if layer is None or layer.shape != shape:
        return np.zeros(shape, dtype=np.uint8)
    return layer

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="エアペイント", page_icon="🎨")
st.title("🎨 エアペイント（Streamlit + MediaPipe Hands）")

with st.sidebar:
    st.header("設定")

    col1, col2 = st.columns(2)
    start = col1.button("🎮 開始")
    stop  = col2.button("🛑 停止")

    color_hex = st.color_picker("色", "#21B1FF")
    brush_size = st.slider("太さ", 3, 40, 10, 1)
    alpha = st.slider("描画の濃さ（重ね透明度）", 0.2, 1.0, 0.9, 0.05)

    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    clear = col3.button("🧹 クリア")
    undo  = col4.button("↩️ アンドゥ")
    save  = col5.button("💾 スナップ保存")

    st.markdown("---")
    pinch_thresh = st.slider("ピンチ判定しきい値(小さいほど敏感)", 0.02, 0.10, 0.05, 0.01)
    show_landmarks = st.checkbox("手の骨格を表示", value=True)

# 状態
if "running" not in st.session_state:
    st.session_state.running = False
if "draw_layer" not in st.session_state:
    st.session_state.draw_layer = None  # BGR
if "strokes" not in st.session_state:
    st.session_state.strokes = []  # 各要素: {"points":[(x,y),...], "color":(B,G,R), "size":int}
if "current_stroke" not in st.session_state:
    st.session_state.current_stroke = None
if "last_save_path" not in st.session_state:
    st.session_state.last_save_path = None
if "info_msg" not in st.session_state:
    st.session_state.info_msg = ""

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# クリア・アンドゥ・保存
def rebuild_layer_from_strokes(base_shape):
    layer = np.zeros(base_shape, dtype=np.uint8)
    for s in st.session_state.strokes:
        pts = s["points"]
        color = s["color"]
        size = s["size"]
        for i in range(1, len(pts)):
            cv2.line(layer, pts[i-1], pts[i], color, size, cv2.LINE_AA)
    return layer

if clear:
    st.session_state.strokes = []
    if st.session_state.draw_layer is not None:
        st.session_state.draw_layer[:] = 0

if undo and len(st.session_state.strokes) > 0 and st.session_state.draw_layer is not None:
    st.session_state.strokes.pop()
    st.session_state.draw_layer = rebuild_layer_from_strokes(st.session_state.draw_layer.shape)

video_placeholder = st.empty()
hint_placeholder = st.empty()

# ------------------------------
# カメラ & MediaPipe Hands
# ------------------------------
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("カメラが見つかりません。接続を確認してください。")
        st.stop()

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        prev_draw = False  # 直前フレームで描画モードだったか
        color_bgr = hex_to_bgr(color_hex)

        while st.session_state.running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            # ミラー表示
            frame = cv2.flip(frame, 1)

            # レイヤー初期化
            st.session_state.draw_layer = ensure_layer(st.session_state.draw_layer, frame.shape)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            draw_now = False
            index_pos = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                if show_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,128,255), thickness=1))

                # ピンチ距離（親指先x人差し指先）で描画ON/OFF
                lm = hand_landmarks.landmark
                ix, iy = lm[8].x, lm[8].y     # 人差し指先
                tx, ty = lm[4].x, lm[4].y     # 親指先

                # 正規化座標で距離計算（0～√2）
                pinch_dist = ((ix - tx)**2 + (iy - ty)**2) ** 0.5
                draw_now = (pinch_dist < pinch_thresh)

                # 画面座標へ変換
                h, w = frame.shape[:2]
                cx, cy = int(ix * w), int(iy * h)
                index_pos = (cx, cy)

                # 現在のモードテキスト
                state_txt = "DRAWING" if draw_now else "MOVE"
                draw_text_with_bg(frame, f"MODE: {state_txt}  (pinch={pinch_dist:.3f})", pos=(20, 40))

                # インデックスの位置目印
                cv2.circle(frame, index_pos, 6, (0, 255, 255), -1)

            else:
                draw_text_with_bg(frame, "手が見えていません", pos=(20, 40), bg=(60,60,60))

            # ストローク記録 & レイヤー描画
            if draw_now and index_pos is not None:
                if not prev_draw:
                    # 新しいストローク開始
                    st.session_state.current_stroke = {
                        "points": [index_pos],
                        "color": hex_to_bgr(st.session_state.get('color_hex', color_hex)),
                        "size": st.session_state.get('brush_size', brush_size)
                    }
                    prev_draw = True
                else:
                    # 既存ストロークに追加
                    st.session_state.current_stroke["points"].append(index_pos)
                    pts = st.session_state.current_stroke["points"]
                    c = st.session_state.current_stroke["color"]
                    s = st.session_state.current_stroke["size"]
                    # 直前点と現在点を結ぶ
                    if len(pts) >= 2:
                        cv2.line(st.session_state.draw_layer, pts[-2], pts[-1], c, s, cv2.LINE_AA)
            else:
                # 描画OFFで、直前が描画中だったらストロークを完了
                if prev_draw and st.session_state.current_stroke is not None:
                    st.session_state.strokes.append(st.session_state.current_stroke)
                    st.session_state.current_stroke = None
                prev_draw = False

            # レイヤーとフレーム合成
            out = cv2.addWeighted(frame, 1.0, st.session_state.draw_layer, alpha, 0)
            video_placeholder.image(out, channels="BGR", use_container_width=True)

            # 保存要求の処理（UIはループ外で作っているので毎フレーム確認）
            if save:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = f"/mnt/data/airpaint_{ts}.png"
                cv2.imwrite(path, out)
                st.session_state.last_save_path = path
                st.success(f"保存しました: {os.path.basename(path)}")
                # 二重保存を防ぐためワンショット扱い
                save = False

    cap.release()
    cv2.destroyAllWindows()
else:
    hint_placeholder.info("左の「🎮 開始」でカメラを起動。親指と人差し指をつまむと描けます。")

# 直近の保存リンク
if st.session_state.last_save_path:
    st.markdown(f"[直近の保存画像を開く]({st.session_state.last_save_path})")
