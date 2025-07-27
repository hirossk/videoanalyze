import streamlit as st
import cv2
import mediapipe as mp
import numpy as np 

# MediaPipeの各ソリューションを準備
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh # ◀ Face Mesh


# --- Streamlit UIの設定 ---
st.title("📹 リアルタイムAI解析アプリを作ろう！")
st.sidebar.markdown("### 解析モードを選択してください")

if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

# ボタンでモードを切り替える
if st.sidebar.button("✨ 顔の特徴 (メッシュ)"): # ◀ Face Mesh
    st.session_state['mode'] = 'FaceMesh'
if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'

st.sidebar.markdown(f"**現在のモード:** `{st.session_state['mode']}`")

# --- メイン処理 ---
frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)

# MediaPipeモデルの読み込み
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh: # ◀ Face Mesh

    while cap.isOpened() and st.session_state['mode'] != 'Stop':
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = image
        # ▼▼▼ Face Meshの処理 ▼▼▼
        if st.session_state['mode'] == 'FaceMesh': # ◀ Face Mesh
                results = face_mesh.process(image_rgb)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 顔の網目を描画
                        mp_drawing.draw_landmarks(
                            image=processed_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

        frame_placeholder.image(processed_image, channels="BGR")

cap.release()
cv2.destroyAllWindows()