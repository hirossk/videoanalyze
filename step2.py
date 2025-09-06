# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2

# Webページに「看板（タイトル）」を出す命令
st.title("📷 シンプルカメラアプリ")

# --- 状態を記憶する「いれもの」の準備 ---

# プログラムが初めて動くときだけ、
# 「run」という名前の「状態いれもの」を作り、「止まっている(False)」という印を入れておく
if 'run' not in st.session_state:
    st.session_state.run = False


# --- ボタンを２つ作る ---

# 「カメラ起動」ボタン。押されると、「状態いれもの(run)」にTrue(「動かす」の合図)を入れる
if st.button('カメラ起動'):
    st.session_state.run = True

# 「カメラ停止」ボタン。押されると、「状態いれもの(run)」にFalse(「止める」の合図)を入れる
if st.button('カメラ停止'):
    st.session_state.run = False


# --- カメラの状態によって、表示を変える ---

# もし「状態いれもの」の中身がTrue(「動かす」の合図)だったら…
if st.session_state.run:
    # 映像を表示するための「空の場所」を確保
    frame_placeholder = st.empty()
    # PCのカメラを起動する
    cap = cv2.VideoCapture(0)

    # 「状態いれもの」がTrueである間、ずっと繰り返す
    while st.session_state.run:
        # カメラから1枚の画像(フレーム)を読み込む
        success, frame = cap.read()
        if not success:
            break

        # 画像を白黒（グレースケール）に変換
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 画像を上下反転
        # frame = cv2.flip(frame, 0)
        # 「空の場所」に、カメラの画像を表示する
        frame_placeholder.image(frame, channels="BGR")
            
    # （ループが終わったら）カメラを解放する（使い終わったら片付ける）
    cap.release()

# もし「状態いれもの」の中身がFalse(「止める」の合図)だったら…
else:
    # メッセージを表示する
    st.write("「カメラ起動」ボタンを押してください。")