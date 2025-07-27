import streamlit as st
import cv2
import numpy as np 
# --- ステップ2で解除 ---
import mediapipe as mp
# --- ステップ4で解除 ---
from ultralytics import YOLO 

# --- ステップ2で解除 ---
# MediaPipeの各ソリューションを準備
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# 

# --- ステップ1から使う ---
st.title("📹 リアルタイムAI解析アプリを作ろう！")
st.sidebar.markdown("### 解析モードを選択してください")

# Streamlitのセッション状態でモードを管理
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

if st.sidebar.button("🖐️ 手の検出 (MediaPipe)"):
    st.session_state['mode'] = 'Hands'

if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'
st.sidebar.markdown(f"**現在のモード:** `{st.session_state['mode']}`")

# --- ステップ1から使う ---
frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)


# --- ステップ2で解除 ---
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#      

    # --- ステップ1から使う ---
    while cap.isOpened() and st.session_state['mode'] != 'Stop':
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        processed_image = image # 処理後の画像を入れる変数

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if st.session_state['mode'] == 'Hands':
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(processed_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        # --- ステップ1から使う ---
        frame_placeholder.image(processed_image, channels="BGR")

# --- ステップ1から使う ---
cap.release()
cv2.destroyAllWindows()
if st.session_state['mode'] == 'Stop':
    st.success("処理を停止しました。")