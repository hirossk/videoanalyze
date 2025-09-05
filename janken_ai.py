
# -*- coding: utf-8 -*-
"""
Streamlit + MediaPipe で作る じゃんけんAI
使い方:
  1) 必要なライブラリをインストール
     pip install streamlit opencv-python mediapipe

  2) 実行
     streamlit run janken_ai.py
"""
import time
import random
from collections import deque, Counter

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# ------------------------------
# ヘルパ関数
# ------------------------------
def classify_hand_gesture(hand_landmarks, image_shape):
    """
    MediaPipe Hands のランドマークからじゃんけんの手を推定する。
    ここでは親指を使わず、
    - 人差し指, 中指, 薬指, 小指 の4本の「伸びている指の本数」で判定します。
      * 0本 or 1本 … GU（グー）
      * 2本（人差し指+中指 っぽい）… CHOKI（チョキ）
      * 3本以上 … PA（パー）
    簡易ヒューリスティクスのため、完全ではありません。
    """
    h, w = image_shape[:2]

    # MediaPipeのインデックス
    TIP_IDS = [8, 12, 16, 20]         # index, middle, ring, pinky
    PIP_IDS = [6, 10, 14, 18]         # それぞれのPIP関節

    finger_up = 0

    # 指先のy座標がPIPよりも上(= 画面座標系では数値が小さい)なら「伸びている」とみなす
    for tip_id, pip_id in zip(TIP_IDS, PIP_IDS):
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        tip_y = int(tip.y * h)
        pip_y = int(pip.y * h)
        if tip_y < pip_y:
            finger_up += 1

    # 分類ルール
    if finger_up >= 3:
        return "PA"
    elif finger_up == 2:
        return "CHOKI"
    elif finger_up <= 1:
        return "GU"
    return "UNKNOWN"


def decide_winner(player, ai):
    """勝敗判定"""
    if player == ai:
        return "DRAW"
    wins = {("GU", "CHOKI"), ("CHOKI", "PA"), ("PA", "GU")}
    return "WIN" if (player, ai) in wins else "LOSE"


def ai_choose(player_guess, mode="FAIR"):
    """
    AIの手を選ぶ。
      FAIR: 完全ランダム
      SLY : 50%でプレイヤーに勝つ手、50%でランダム
      CHEAT: 常にプレイヤーに勝つ手（デモ用）
    """
    choices = ["GU", "PA", "CHOKI"]
    if mode == "FAIR" or player_guess is None:
        return random.choice(choices)

    # プレイヤーに勝つ手
    beats = {"GU": "PA", "PA": "CHOKI", "CHOKI": "GU"}
    if mode == "SLY":
        return beats[player_guess] if random.random() < 0.5 else random.choice(choices)
    if mode == "CHEAT":
        return beats[player_guess]
    return random.choice(choices)


def draw_overlay_text(img, text, pos=(30, 50), color=(0, 255, 0), scale=1.2, thickness=2):
    """OpenCVで読みやすい帯付きテキストを描画"""
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x-10, y-h-10), (x + w + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return img


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="じゃんけんAI", page_icon="✊")
st.title("✊✌️🖐️ じゃんけんAI（Streamlit + MediaPipe Hands）")

with st.sidebar:
    st.header("設定")
    ai_mode = st.selectbox("AIの強さ", ["FAIR（ランダム）", "SLY（たまにズル）", "CHEAT（常にズル）"], index=0)
    ai_mode_key = {"FAIR（ランダム）": "FAIR", "SLY（たまにズル）": "SLY", "CHEAT（常にズル）": "CHEAT"}[ai_mode]
    round_seconds = st.slider("カウントダウン秒", 2, 5, 3, 1)
    sample_frames = st.slider("判定の安定化(直近Nフレーム多数決)", 3, 15, 7, 1)

    col_btn1, col_btn2 = st.columns(2)
    start_clicked = col_btn1.button("🎮 ゲーム開始")
    stop_clicked  = col_btn2.button("🛑 停止")

    reset_score = st.button("🔁 スコアリセット")

# 状態管理
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
    st.session_state.countdown_end = None
    st.session_state.buff.clear()

if stop_clicked:
    st.session_state.playing = False

# スコア表示
sc1, sc2, sc3 = st.columns([1,1,2])
sc1.metric("あなた", st.session_state.score_player)
sc2.metric("AI", st.session_state.score_ai)
sc3.write(f"**直近の結果:** {st.session_state.last_result}")

# プレースホルダ
video_placeholder = st.empty()
info_placeholder = st.empty()

# ------------------------------
# カメラ & MediaPipe Hands
# ------------------------------
if st.session_state.playing:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("カメラが見つかりません。外付けWebカメラの場合は接続を確認してください。")
        st.stop()

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        # ラウンド管理
        round_locked = False
        player_choice = None
        ai_choice = None
        show_result_until = 0.0

        # カウントダウン初期化
        if st.session_state.countdown_end is None:
            st.session_state.countdown_end = time.time() + round_seconds

        while st.session_state.playing and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            # ミラー表示
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # すでに勝敗表示中なら、そのまま表示して時間経過で次のラウンドへ
            now = time.time()
            if now < show_result_until:
                # 直前の勝敗オーバーレイ
                txt = f"YOU: {player_choice or '-'}   AI: {ai_choice or '-'}   => {st.session_state.last_result}"
                frame = draw_overlay_text(frame, txt, pos=(20, 60), color=(0,255,255), scale=0.9, thickness=2)
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
                continue
            else:
                round_locked = False
                player_choice = None
                ai_choice = None

            # 手検出
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # ジェスチャ分類
                    gesture = classify_hand_gesture(hand_landmarks, frame.shape)
                    if gesture != "UNKNOWN":
                        st.session_state.buff.append(gesture)

                    # 手の骨格を軽く描く（デモ性向上。重い場合はコメントアウト）
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,128,255), thickness=1))

            # カウントダウン
            sec_left = int(st.session_state.countdown_end - time.time())
            if sec_left >= 1:
                frame = draw_overlay_text(frame, f"{sec_left}", pos=(30, 120), color=(255,255,0), scale=3.0, thickness=6)
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
                continue

            # 0 になった: ラウンドをロックして判定
            if not round_locked:
                round_locked = True

                # 直近Nフレームの最多出現ラベルを採用
                recent = list(st.session_state.buff)[-sample_frames:]
                player_choice = Counter(recent).most_common(1)[0][0] if recent else None

                # AIの手を決定
                ai_choice = ai_choose(player_choice, ai_mode_key)

                # 勝敗
                if player_choice is None:
                    st.session_state.last_result = "NO HAND"
                else:
                    res = decide_winner(player_choice, ai_choice)
                    if res == "WIN":
                        st.session_state.score_player += 1
                    elif res == "LOSE":
                        st.session_state.score_ai += 1
                    st.session_state.last_result = res

                # 結果表示を1.2秒ほど
                show_result_until = time.time() + 1.2

                # 次ラウンドのカウントダウン再セット
                st.session_state.countdown_end = time.time() + round_seconds
                st.session_state.buff.clear()

            # 結果オーバーレイ
            txt = f"YOU: {player_choice or '-'}   AI: {ai_choice or '-'}   => {st.session_state.last_result}"
            frame = draw_overlay_text(frame, txt, pos=(20, 60), color=(0,255,255), scale=0.9, thickness=2)
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()
else:
    info_placeholder.info("左の「🎮 ゲーム開始」を押すとカメラが起動し、カウントダウン後に判定します。")
    st.write("学びのポイント：手のランドマーク座標から **指が伸びているか/曲がっているか** を判定して、じゃんけんにマッピングしています。")
