# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2
import mediapipe as mp
# 自分たちで作った「専門家」のファイルをインポートする
from processors import pose_counter

# ページ全体の設定。layout="wide" で画面を広く使う（最初のStreamlit命令にするのがルール）
st.set_page_config(page_title="ストレッチ回数カウンター", page_icon="🤸", layout="wide")

# AIモデル（姿勢の専門家）を最初に1回だけ読み込んで、
# 賢く使いまわすための工夫（キャッシュ機能）
@st.cache_resource
def load_pose_model():
    # MediaPipeから「姿勢を見つける専門家」を準備する
    return mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- ここからWebページの見た目を作る ---
st.title("🤸 ストレッチ回数カウンター")
st.caption("AI（MediaPipe）で姿勢を検出し、腕のストレッチ回数を自動でカウントします")

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

# with st.sidebar: と書くと、この中の命令がぜんぶ左側のサイドバーに表示される
with st.sidebar:
    st.header("🎛️ コントロール")
    st.caption("ボタンでモードを切り替えます")

    # 「ストレッチ」ボタン。押されたら、メモ帳の「mode」を「Pose」に書き換える
    # width="stretch" でボタンをサイドバーの幅いっぱいに広げる
    if st.button("🤸‍♀️ ストレッチ開始", width="stretch", type="primary"):
        st.session_state['mode'] = 'Pose'
        st.session_state['counter'] = 0 # カウンターをリセット

    # 「停止」ボタン。押されたら、メモ帳の「mode」を「Stop」に書き換える
    if st.button("🛑 停止", width="stretch"):
        st.session_state['mode'] = 'Stop'

    st.divider()

    # st.metric() は、大きな数字をカッコよく見せるモダンな命令
    # st.empty() で「空の場所」を作っておき、あとでループの中から回数を上書き更新する
    count_metric = st.empty()
    count_metric.metric("🤸 ストレッチ回数", st.session_state['counter'])

    # st.badge() で、今の状態を色つきラベルでわかりやすく表示する
    if st.session_state['mode'] == 'Stop':
        st.badge("停止中", icon="⚪", color="grey")
    else:
        st.badge("計測中", icon="🟢", color="green")

# --- ここからカメラの映像を処理する ---

# 映像を画面幅の約80%に収めるため、中央のカラムに表示する（左右に余白を作る [1:8:1]）
_, _col_video, _ = st.columns([1, 8, 1])
frame_placeholder = _col_video.empty()
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
    if st.session_state['mode'] == 'Pose':
        # 専門家に「今のカメラ画像」と「メモ帳の内容」を渡して、処理をお願いする
        processed_image, new_counter, new_stage, prev_current = pose_counter.process(
            image, pose_model, st.session_state['counter'], st.session_state['stage'], prev_current)
        
        # 専門家から返ってきた新しい情報で、メモ帳を更新する
        st.session_state['counter'] = new_counter
        st.session_state['stage'] = new_stage

        # サイドバーの「ストレッチ回数」を、最新の数字にリアルタイムで上書き更新する
        count_metric.metric("🤸 ストレッチ回数", st.session_state['counter'])

    # 準備しておいた「空の場所（額縁）」に、処理が終わった画像を表示する
    # width="stretch" で映像を画面の幅いっぱいに広げる
    frame_placeholder.image(processed_image, channels="BGR", width="stretch")

# （ループが終わったら）使い終わったカメラを解放する（お片付け）
cap.release()
cv2.destroyAllWindows()