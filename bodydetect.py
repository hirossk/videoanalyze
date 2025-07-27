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
mp_face_detection = mp.solutions.face_detection
# --- ステップ3で解除 ---
# mp_hands = mp.solutions.hands
# --- ステップ5で解除 ---
# mp_pose = mp.solutions.pose 
# mp_selfie_segmentation = mp.solutions.selfie_segmentation

# --- ステップ4で解除 ---
# YOLOv8モデルの読み込み（キャッシュを利用して高速化）
# @st.cache_resource
# def load_yolo_model():
#     model = YOLO('yolov8n.pt')
#     return model

# --- ステップ1から使う ---
st.title("📹 リアルタイムAI解析アプリを作ろう！")
st.sidebar.markdown("### 解析モードを選択してください")

# Streamlitのセッション状態でモードを管理
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

# --- ステップ2で解除 ---
# if st.sidebar.button("🙂 顔の検出 (MediaPipe)"):
#     st.session_state['mode'] = 'Face'
# --- ステップ3で解除 ---
# if stsidebar.button("🖐️ 手の検出 (MediaPipe)"):
#     st.session_state['mode'] = 'Hands'
# --- ステップ4で解除 ---
# if st.sidebar.button("📦 物体検出 (YOLOv8)"):
#     st.session_state['mode'] = 'YOLOv8'
# --- ステップ5で解除 ---
# if st.sidebar.button("🕺 全身の姿勢推定 (MediaPipe)"): 
#     st.session_state['mode'] = 'Pose'
# if st.sidebar.button("🖼️ 背景をぼかす (MediaPipe)"): 
#     st.session_state['mode'] = 'Segmentation'
# --- ステップ1から使う ---
if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'
st.sidebar.markdown(f"**現在のモード:** `{st.session_state['mode']}`")

# --- ステップ1から使う ---
frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)

# --- ステップ4で解除 ---
# yolo_model = load_yolo_model()

# --- ステップ2で解除 ---
# with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
#      mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
#      mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
#      mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:

    # --- ステップ1から使う ---
while cap.isOpened() and st.session_state['mode'] != 'Stop':
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        processed_image = image # 処理後の画像を入れる変数

        # --- ステップ4で解除 ---
        # ▼▼▼ YOLOv8の処理 ▼▼▼
        if st.session_state['mode'] == 'YOLOv8':
            pass
            # results = yolo_model.predict(image, verbose=False)
            # processed_image = results[0].plot()
        # --- ステップ2で解除 ---
        # ▼▼▼ MediaPipeの処理 ▼▼▼
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # if st.session_state['mode'] == 'Face':
            #     results = face_detection.process(image_rgb)
            #     if results.detections:
            #         for detection in results.detections:
            #             mp_drawing.draw_detection(processed_image, detection)
            # --- ステップ3で解除 ---
            # elif st.session_state['mode'] == 'Hands':
            #     results = hands.process(image_rgb)
            #     if results.multi_hand_landmarks:
            #         for hand_landmarks in results.multi_hand_landmarks:
            #             mp_drawing.draw_landmarks(processed_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # --- ステップ5で解除 ---
            # elif st.session_state['mode'] == 'Pose':
            #     results = pose.process(image_rgb)
            #     if results.pose_landmarks:
            #         mp_drawing.draw_landmarks(processed_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # elif st.session_state['mode'] == 'Segmentation':
            #     results = selfie_segmentation.process(image_rgb)
            #     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            #     bg_image = cv2.GaussianBlur(processed_image, (55, 55), 0)
            #     processed_image = np.where(condition, processed_image, bg_image)

        # --- ステップ1から使う ---
        frame_placeholder.image(processed_image, channels="BGR")

# --- ステップ1から使う ---
cap.release()
cv2.destroyAllWindows()
if st.session_state['mode'] == 'Stop':
    st.success("処理を停止しました。")