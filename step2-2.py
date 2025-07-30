# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# MediaPipeという道具箱から、「人と背景を見分ける専門家」を準備
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# 人と背景の判定に使うしきい値（0.0～1.0）
SEGMENTATION_THRESHOLD = 0.1

# --- ここからWebページの見た目を作る ---

# Webページに一番大きな「看板（タイトル）」を出す
st.title("🖼️ 背景ぼかしアプリ")
# 画面の左側（サイドバー）に「説明」を表示する
st.sidebar.markdown("モードを選択してください")


# --- ボタンが押されたときの「状態」を覚えておく仕組み ---

# もし「mode」という名前の「状態いれもの」がなければ、最初に作っておく
# 最初は「止まっている(Stop)」状態にしておく
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

# 「背景をぼかす」ボタン。押されたら、「状態いれもの」に「Segmentation」という文字を入れる
if st.sidebar.button('🖼️ 背景をぼかす'):
    st.session_state['mode'] = 'Segmentation'

# 「停止」ボタン。押されたら、「状態いれもの」に「Stop」という文字を入れる
if st.sidebar.button('🛑 停止'):
    st.session_state['mode'] = 'Stop'


# --- ここからカメラの映像を処理する ---

# 映像を表示するための「空の場所（額縁）」をページに用意する
frame_placeholder = st.empty()

# もし今のモードが「Segmentation」だったら、カメラを起動して処理を始める
if st.session_state['mode'] == 'Segmentation':
    # PCのカメラを起動する
    cap = cv2.VideoCapture(0)
    
    # 「人と背景を見分ける専門家」を呼び出して、準備してもらう
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        
        # 「停止」ボタンが押されるまで、ずっと繰り返す
        while cap.isOpened() and st.session_state['mode'] == 'Segmentation':
            # カメラから1枚の画像(フレーム)を読み込む
            success, image = cap.read()
            if not success:
                break

            # 映像を鏡のように左右反転させる
            image = cv2.flip(image, 1)
            # 処理した後の画像を入れるための変数を用意
            processed_image = image.copy()
            
            # 画像の色を、AIが理解しやすい「RGB」形式に変換する
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 「専門家」に画像を見せて、「これは人」「これは背景」という風に見分けてもらう
            results = selfie_segmentation.process(image_rgb)
            
            # 「ここが人ですよ」という情報(マスク)を元に、背景だけをぼかす
            # np.stackで、白黒のマスクをカラー画像と同じ3次元に変換する
            # 計算式の意味は？
            # マスクの値が0.1より大きい部分（人の部分）をTrue、それ以外をFalseにする
            # これで、マスクが人の部分はTrue、背景はFalseになる
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > SEGMENTATION_THRESHOLD
            
            # 元の画像をぼかして、ぼやけた背景画像を作る
            bg_image = cv2.GaussianBlur(processed_image, (55, 55), 0)
            
            # np.whereで、「人」の部分は元の画像を、「背景」の部分はぼかした画像を合成する
            processed_image = np.where(condition, processed_image, bg_image)
            
            # 準備しておいた「空の場所（額縁）」に、処理が終わった画像を表示する
            frame_placeholder.image(processed_image, channels="BGR")
            
    # （ループが終わったら）使い終わったカメラを解放する（お片付け）
    cap.release()

# もし今のモードが「Stop」だったら…
else:
    # メッセージを表示する
    frame_placeholder.write("「背景をぼかす」ボタンを押してください。")