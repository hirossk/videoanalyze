
# -*- coding: utf-8 -*-
import time
import random
from collections import deque, Counter

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp


import janken_logic  # 判定ロジック

ROUND_SECONDS = 4
SAMPLE_FRAMES = 3

def ai_choose():
    return random.choice(["GU", "PA", "CHOKI"])

def draw_overlay_text(img, text, pos=(30, 50), color=(0, 255, 0), scale=1.2, thickness=2):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x-10, y-h-10), (x + w + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return img

st.set_page_config(page_title="シンプルじゃんけんAI(修正版)", page_icon="✊", layout="wide")
st.title("✊✌️🖐️ シンプルじゃんけんAI（修正版）")

with st.sidebar:
    st.header("コントロール")
    start_clicked = st.button("🎮 ゲーム開始")
    stop_clicked  = st.button("🛑 停止")
    reset_score = st.button("🔁 スコアリセット")

# 状態
if "playing" not in st.session_state:
    st.session_state.playing = False
if "score_player" not in st.session_state:
    st.session_state.score_player = 0
if "score_ai" not in st.session_state:
    st.session_state.score_ai = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = "-"
if "countdown_end" not in st.session_state:
    st.session_state.countdown_end = None
if "buff" not in st.session_state:
    st.session_state.buff = deque(maxlen=20)

if reset_score:
    st.session_state.score_player = 0
    st.session_state.score_ai = 0
    st.session_state.last_result = "-"

if start_clicked:
    st.session_state.playing = True
    st.session_state.countdown_end = time.time() + ROUND_SECONDS
    st.session_state.buff.clear()

if stop_clicked:
    st.session_state.playing = False

# スコア用プレースホルダ（ループ内で更新する）
col1, col2, col3 = st.columns([1, 1, 2])
p_metric_ph = col1.empty()
a_metric_ph = col2.empty()
res_text_ph = col3.empty()

# ビデオ表示プレースホルダ
# 映像を画面幅の約80%に収めるため、中央のカラムに表示する（左右に余白を作る [1:8:1]）
_, _col_video, _ = st.columns([1, 8, 1])
video_placeholder = _col_video.empty()
info_placeholder = st.empty()

def update_scoreboard():
    p_metric_ph.metric("あなた", st.session_state.score_player)
    a_metric_ph.metric("AI", st.session_state.score_ai)
    res_text_ph.write(f"**直近の結果:** {st.session_state.last_result}")

update_scoreboard()

if st.session_state.playing:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("カメラが見つかりません。別プロセスがカメラを使用していないか確認してください。")
        st.stop()

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        model_complexity=0, max_num_hands=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    ) as hands:

        round_locked = False
        player_choice = None
        ai_choice = None
        show_result_until = 0.0

        # 初回のみ防御的にカウントダウン設定
        if st.session_state.countdown_end is None:
            st.session_state.countdown_end = time.time() + ROUND_SECONDS

        while st.session_state.playing and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time()

            # 勝敗表示中
            if now < show_result_until:
                txt = f"YOU: {player_choice or '-'}   AI: {ai_choice or '-'}   => {st.session_state.last_result}"
                frame = draw_overlay_text(frame, txt, pos=(20, 60), color=(0,255,255), scale=0.9, thickness=2)
                video_placeholder.image(frame, channels="BGR", width="stretch")
                time.sleep(0.01)
                continue
            else:
                round_locked = False

            # 手検出
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = janken_logic.classify_hand_gesture(hand_landmarks, frame.shape)
                    if gesture != "UNKNOWN":
                        st.session_state.buff.append(gesture)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,128,255), thickness=1))

            # カウントダウン
            sec_left = int(st.session_state.countdown_end - time.time())
            if sec_left >= 1:
                frame = draw_overlay_text(frame, f"{sec_left}", pos=(30, 120), color=(255,255,0), scale=3.0, thickness=6)
                video_placeholder.image(frame, channels="BGR", width="stretch")
                time.sleep(0.01)
                continue

            # 判定
            if not round_locked:
                round_locked = True
                recent = list(st.session_state.buff)[-SAMPLE_FRAMES:]
                player_choice = Counter(recent).most_common(1)[0][0] if recent else None
                ai_choice = ai_choose()

                if player_choice is None:
                    st.session_state.last_result = "NO HAND"
                else:
                    res = janken_logic.decide_winner(player_choice, ai_choice)
                    if res == "WIN":
                        st.session_state.score_player += 1
                    elif res == "LOSE":
                        st.session_state.score_ai += 1
                    st.session_state.last_result = res

                # スコア表示を更新（rerunせずに反映）
                update_scoreboard()

                show_result_until = time.time() + 1.2
                st.session_state.countdown_end = time.time() + ROUND_SECONDS
                st.session_state.buff.clear()

            video_placeholder.image(frame, channels="BGR", width="stretch")
            # ループ回りすぎ抑制
            time.sleep(0.005)

    cap.release()
    cv2.destroyAllWindows()
else:
    info_placeholder.info("左の「🎮 ゲーム開始」を押すとカメラが起動します。")
