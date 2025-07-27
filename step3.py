import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# MediaPipeの各ソリューションを準備
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

# Streamlit UI
st.title("📹 リアルタイムAI解析アプリを作ろう！")
st.sidebar.markdown("### 解析モードを選択してください")

# セッション状態でモードを管理
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

if st.sidebar.button("🙂 顔の検出"):
    st.session_state['mode'] = 'Face'

if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'

st.sidebar.markdown(f"**現在のモード:** `{st.session_state['mode']}`")

frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened() and st.session_state['mode'] != 'Stop':
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        processed_image = image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if st.session_state['mode'] == 'Face':
            results = face_detection.process(image_rgb)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(processed_image, detection)

        frame_placeholder.image(processed_image, channels="BGR")

cap.release()
cv2.destroyAllWindows()

if st.session_state['mode'] == 'Stop':
    st.success("処理を停止しました。")
