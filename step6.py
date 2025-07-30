# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2
# 自分たちで作った「カートゥーン風加工の専門家」のファイルをインポートする
from processors import cartoon_styler

# Webページに「看板（タイトル）」を出す
st.title("📹 カートゥーン風エフェクトデモ")
# 画面の左側（サイドバー）に「説明」を表示する
st.sidebar.markdown("モードを選択してください")


# --- アプリの状態を覚えておくための「メモ帳」の準備 ---
# st.session_state というアプリ専用のメモ帳を使う

# もしメモ帳に「mode」という項目がなければ、最初に「Stop」と書いておく
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'


# --- ボタンを作って、押されたらメモ帳を書き換える ---

# 「カートゥーン風エフェクト」ボタン。押されたら、メモ帳の「mode」を「Cartoon」に書き換える
if st.sidebar.button("🎨 カートゥーン風エフェクト"):
    st.session_state['mode'] = 'Cartoon'
# 「停止」ボタン。押されたら、メモ帳の「mode」を「Stop」に書き換える
if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'


# --- エフェクトの強さを調整する「つまみ（スライダー）」を作る ---

# もし今のモードが「Cartoon」だったら、サイドバーに調整用のスライダーを表示する
if st.session_state['mode'] == 'Cartoon':
    st.sidebar.subheader("パラメータ調整")
    # st.sidebar.slider() で、見た目を調整する「つまみ」を作る
    bilateral_d = 7
    bilateral_sigmaColor = 450
    median_ksize = 3
    adaptive_blockSize = 9
    adaptive_C = 5
    # bilateral_d = st.sidebar.slider("色の滑らかさ (d)", 3, 15, 7, step=2)
    # bilateral_sigmaColor = st.sidebar.slider("色の範囲 (sigmaColor)", 50, 500, 450, step=10)
    # median_ksize = st.sidebar.slider("輪郭の滑らかさ (ksize)", 3, 15, 3, step=2)
    # adaptive_blockSize = st.sidebar.slider("輪郭の細かさ (blockSize)", 3, 25, 9, step=2)
    # adaptive_C = st.sidebar.slider("輪郭の強さ (C)", 0, 10, 5, step=1)


# --- ここからカメラの映像を処理する ---

# 映像を表示するための「空の場所（額縁）」をページに用意する
frame_placeholder = st.empty()

# もし今のモードが「Cartoon」だったら、カメラを起動して処理を始める
if st.session_state['mode'] == 'Cartoon':
    # PCのカメラを起動する
    cap = cv2.VideoCapture(0)

    # 「停止」ボタンが押されるまで、ずっと繰り返す
    while st.session_state['mode'] == 'Cartoon':
        # カメラから1枚の画像(フレーム)を読み込む
        success, image = cap.read()
        if not success:
            st.error("カメラの読み込みに失敗しました。")
            break

        # 映像を鏡のように左右反転させる
        image = cv2.flip(image, 1)
        
        # 「専門家」に「今のカメラ画像」と「スライダーで調整した値」を渡して、加工をお願いする
        processed_image = cartoon_styler.process(
            image,
            bilateral_d=bilateral_d,
            bilateral_sigmaColor=bilateral_sigmaColor,
            bilateral_sigmaSpace=bilateral_sigmaColor,
            median_ksize=median_ksize,
            adaptive_blockSize=adaptive_blockSize,
            adaptive_C=adaptive_C
        )
        
        # 準備しておいた「空の場所（額縁）」に、加工が終わった画像を表示する
        frame_placeholder.image(processed_image, channels="BGR")
    
    # （ループが終わったら）使い終わったカメラを解放する（お片付け）
    cap.release()

# もし今のモードが「Stop」だったら…
else:
    # メッセージを表示する
    frame_placeholder.write("「カートゥーン風エフェクト」ボタンを押してください。")