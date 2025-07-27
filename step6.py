import streamlit as st
import cv2
import mediapipe as mp
# --- 専門家をインポート ---
from processors import pose_counter


@st.cache_resource
def load_pose_model():
    return mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- UI設定 ---
st.title("📹 統合リアルタイム解析デモ")
st.sidebar.markdown("### 解析モードを選択してください")

if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'
if 'counter' not in st.session_state:
    st.session_state['counter'] = 0
if 'stage' not in st.session_state:
    st.session_state['stage'] = "水平"

# ボタンでモードを切り替える
if st.sidebar.button("💪 肩のストレッチ"):
    st.session_state['mode'] = 'Shoulder'
    st.session_state['counter'] = 0
if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'

# --- メイン処理 ---
frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)

# モデルをロード
pose_model = load_pose_model()
prev_current = "水平"

while cap.isOpened() and st.session_state['mode'] != 'Stop':
    success, image = cap.read()
    if not success: break
    image = cv2.flip(image, 1)
    
    processed_image = image
    
    # --- 店長が専門家に仕事を依頼 ---
    if st.session_state['mode'] == 'Shoulder':
        processed_image, new_counter, new_stage ,prev_current= pose_counter.process(
            image, pose_model, st.session_state['counter'], st.session_state['stage'], prev_current)
        st.session_state['counter'] = new_counter
        st.session_state['stage'] = new_stage


    frame_placeholder.image(processed_image, channels="BGR")

cap.release()
cv2.destroyAllWindows()