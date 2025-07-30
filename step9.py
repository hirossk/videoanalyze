# 必要なライブラリをインポート
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# アバターの円の大きさと線の太さ（定数として最初に定義）
AVATAR_CIRCLE_RADIUS = 15  # 円の半径
AVATAR_LINE_THICKNESS = 10 # 線の太さ

# MediaPipeの準備
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Streamlit UIのセットアップ ---
st.title("🤖 ポーズで動く！ロボット風アバター")
st.sidebar.markdown("### 解析モードを選択してください")

if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

if st.sidebar.button("🤖 アバター表示"):
    st.session_state['mode'] = 'Avatar'
if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'

# --- カメラ処理 ---
frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened() and st.session_state['mode'] != 'Stop':
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        # アバターを描画するために、元の画像と同じ大きさの真っ黒な画像を用意
        avatar_image = np.zeros(image.shape, dtype=np.uint8)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # --- 各関節に円を描画 ---
            body_parts = [
                mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ]
            for part in body_parts:
                lm = landmarks[part.value]
                cx, cy = int(lm.x * w), int(lm.y * h)
                # cv2.circle(avatar_image, (cx, cy), AVATAR_CIRCLE_RADIUS, (255, 255, 0), -1) # 青緑色の円

            # --- 関節を線で結ぶ ---
            connections = [
                # 胴体
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                # 腕
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                # 脚
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            ]
            for start_node, end_node in connections:
                start_lm = landmarks[start_node.value]
                end_lm = landmarks[end_node.value]
                start_point = (int(start_lm.x * w), int(start_lm.y * h))
                end_point = (int(end_lm.x * w), int(end_lm.y * h))
                # cv2.line(avatar_image, start_point, end_point, (255, 255, 255), AVATAR_LINE_THICKNESS) # 白い太線

        # 元のカメラ画像と、作成したアバター画像を合成
        # 0.7と0.3は画像の透明度。数字を変えると、どちらを濃く表示するかが変わる
        combined_image = cv2.addWeighted(image, 0.7, avatar_image, 0.3, 0)
        
        frame_placeholder.image(combined_image, channels="BGR")

    cap.release()