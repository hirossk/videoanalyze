# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2

# ページ全体の設定。layout="wide" にすると画面いっぱいに広く使えるよ
# （一番最初のStreamlit命令として書くのがルール）
st.set_page_config(page_title="シンプルカメラアプリ", page_icon="📷", layout="wide")

# Webページに「看板（タイトル）」を出す命令
st.title("📷 シンプルカメラアプリ")
# st.caption() は、タイトルの下に小さな説明文を出す命令
st.caption("PCのカメラ映像をブラウザにリアルタイムで表示します")

# --- 状態を記憶する「いれもの」の準備 ---

# プログラムが初めて動くときだけ、
# 「run」という名前の「状態いれもの」を作り、「止まっている(False)」という印を入れておく
if 'run' not in st.session_state:
    st.session_state.run = False


# --- ボタンを２つ作る ---

# st.columns() で画面を横並びのエリアに分けて、ボタンを横に2つ並べる（モダンな見た目）
col_start, col_stop = st.columns(2)

# 「カメラ起動」ボタン。押されると、「状態いれもの(run)」にTrue(「動かす」の合図)を入れる
# width="stretch" でボタンをエリアの幅いっぱいに広げる
if col_start.button('▶️ カメラ起動', width="stretch", type="primary"):
    st.session_state.run = True

# 「カメラ停止」ボタン。押されると、「状態いれもの(run)」にFalse(「止める」の合図)を入れる
if col_stop.button('⏹️ カメラ停止', width="stretch"):
    st.session_state.run = False

# st.badge() は、今の状態を色つきの「ラベル」で見せるモダンな命令
if st.session_state.run:
    st.badge("カメラ起動中", icon="🟢", color="green")
else:
    st.badge("停止中", icon="⚪", color="grey")


# --- カメラの状態によって、表示を変える ---

# もし「状態いれもの」の中身がTrue(「動かす」の合図)だったら…
if st.session_state.run:
    # 映像を画面幅の約80%に収めるため、中央のカラムに表示する（左右に余白を作る [1:8:1]）
    _, _col_video, _ = st.columns([1, 8, 1])
    frame_placeholder = _col_video.empty()
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
        # width="stretch" で映像を画面の幅いっぱいに広げる
        frame_placeholder.image(frame, channels="BGR", width="stretch")
            
    # （ループが終わったら）カメラを解放する（使い終わったら片付ける）
    cap.release()

# もし「状態いれもの」の中身がFalse(「止める」の合図)だったら…
else:
    # st.info() は、青い枠つきの案内メッセージを表示するモダンな命令
    st.info("「▶️ カメラ起動」ボタンを押してください。", icon="📷")