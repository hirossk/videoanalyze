# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2
import mediapipe as mp
# 自分たちで作った「専門家」のファイルをインポートする
from processors import pose_counter

# AIモデル（姿勢の専門家）を最初に1回だけ読み込んで、
# 賢く使いまわすための工夫（キャッシュ機能）
@st.cache_resource
def load_pose_model():
    # MediaPipeから「姿勢を見つける専門家」を準備する
    return mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- ここからWebページの見た目を作る ---
st.title("📹 統合リアルタイム解析デモ")
st.sidebar.markdown("### 解析モードを選択してください")

# --- アプリの状態を覚えておくための「メモ帳」の準備 ---
# st.session_state というアプリ専用のメモ帳を使う

# もしメモ帳に「mode」という項目がなければ、最初に「Stop」と書いておく
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'
# もしメモ帳に「counter」という項目がなければ、最初に「0」と書いておく
if 'counter' not in st.session_state:
    st.session_state['counter'] = 0
# もしメモ帳に「stage」という項目がなければ、最初に「水平」と書いておく
if 'stage' not in st.session_state:
    st.session_state['stage'] = "水平"

# --- ボタンを作って、押されたらメモ帳を書き換える ---

# 「肩のストレッチ」ボタン。押されたら、メモ帳の「mode」を「Shoulder」に書き換える
if st.sidebar.button("💪 肩のストレッチ"):
    st.session_state['mode'] = 'Shoulder'
    st.session_state['counter'] = 0 # カウンターをリセット

# 「停止」ボタン。押されたら、メモ帳の「mode」を「Stop」に書き換える
if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'

# --- ここからカメラの映像を処理する ---

# 映像を表示するための「空の場所（額縁）」をページに用意する
frame_placeholder = st.empty()
# PCのカメラを起動する
cap = cv2.VideoCapture(0)

# 準備しておいたAIモデルをロード
pose_model = load_pose_model()
# １つ前の腕の状態を覚えておくための変数
prev_current = "水平"

# カメラが起動していて、かつメモ帳のモードが「Stop」ではない間、ずっと繰り返す
while cap.isOpened() and st.session_state['mode'] != 'Stop':
    # カメラから1枚の画像(フレーム)を読み込む
    success, image = cap.read()
    if not success: break
    # 映像を鏡のように左右反転させる
    image = cv2.flip(image, 1)
    
    # 処理した後の画像を入れるための変数を用意
    processed_image = image
    
    # --- 「店長」が「専門家」に仕事を依頼する ---
    
    # もし今のモードが「Shoulder」なら、pose_counterの専門家に仕事を任せる
    if st.session_state['mode'] == 'Shoulder':
        # 専門家に「今のカメラ画像」と「メモ帳の内容」を渡して、処理をお願いする
        processed_image, new_counter, new_stage, prev_current = pose_counter.process(
            image, pose_model, st.session_state['counter'], st.session_state['stage'], prev_current)
        
        # 専門家から返ってきた新しい情報で、メモ帳を更新する
        st.session_state['counter'] = new_counter
        st.session_state['stage'] = new_stage

    # 準備しておいた「空の場所（額縁）」に、処理が終わった画像を表示する
    frame_placeholder.image(processed_image, channels="BGR")

# （ループが終わったら）使い終わったカメラを解放する（お片付け）
cap.release()
cv2.destroyAllWindows()