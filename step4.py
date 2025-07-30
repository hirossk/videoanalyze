# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# MediaPipeという道具箱から、「絵を描く道具」と「手を見つける専門家」を準備
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# --- ここからWebページの見た目を作る ---

# Webページに一番大きな「看板（タイトル）」を出す
st.title("📹 リアルタイムAI解析アプリを作ろう！")
# 画面の左側（サイドバー）に「説明」を表示する
st.sidebar.markdown("### 解析モードを選択してください")


# --- ボタンが押されたときの「状態」を覚えておく仕組み ---

# もし「mode」という名前の「状態いれもの」がなければ、最初に作っておく
# 最初は「止まっている(Stop)」状態にしておく
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

# 「手の検出」ボタン。押されると、「状態いれもの」に「Hands」という文字を入れる
if st.sidebar.button("🖐️ 手の検出 (MediaPipe)"):
    st.session_state['mode'] = 'Hands'

# 「停止」ボタン。押されると、「状態いれもの」に「Stop」という文字を入れる
if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'

# 今の状態（モード）をサイドバーに表示する
st.sidebar.markdown(f"**現在のモード:** `{st.session_state['mode']}`")


# --- ここからカメラの映像を処理する ---

# 映像を表示するための「空の場所（額縁）」をページに用意する
frame_placeholder = st.empty()
# PCのカメラを起動する
cap = cv2.VideoCapture(0)

# 「手を見つける専門家」を呼び出して、準備してもらう
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    # カメラが起動していて、かつ「停止」モードではない間、ずっと繰り返す
    while cap.isOpened() and st.session_state['mode'] != 'Stop':
        # カメラから1枚の画像(フレーム)を読み込む
        success, image = cap.read()
        if not success:
            break

        # 映像を鏡のように左右反転させる
        image = cv2.flip(image, 1)
        # 処理した後の画像を入れるための変数を用意
        processed_image = image.copy()
        # 描画エリアを真っ暗にする 黒は0
        # processed_image[:] = 0

        # 画像の色を、AIが理解しやすい「RGB」形式に変換する
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # もし今のモードが「Hands」なら、手の検出を行う
        if st.session_state['mode'] == 'Hands':
            # 「手を見つける専門家」に画像を見せて、手を探してもらう
            results = hands.process(image_rgb)
            
            # もし手が見つかったら、その場所に骨格を描く
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # 手の骨格を描く
                    # mp_drawing.draw_landmarks(processed_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # 各指のランドマークインデックス
                    # finger_names = ['', '', '', '', '']
                    # finger_tips = [4, 8, 12, 16, 20]
                    # h, w, _ = processed_image.shape
                    # for name, tip_idx in zip(finger_names, finger_tips):
                    #     tip = hand_landmarks.landmark[tip_idx]
                    #     x, y = int(tip.x * w), int(tip.y * h)
                    #     cv2.putText(processed_image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                # 左右の手のランドマークをそれぞれ描く
                # for idx, handedness in enumerate(results.multi_handedness):
                #     label = handedness.classification[0].label  # 'Left' or 'Right'
                #     hand_landmarks = results.multi_hand_landmarks[idx]
                #     h, w, _ = processed_image.shape
                #     # 手首の位置に左右ラベルを描画
                #     wrist = hand_landmarks.landmark[0]
                #     x, y = int(wrist.x * w), int(wrist.y * h)
                #     cv2.putText(processed_image, label, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
                    pass
        # 準備しておいた「空の場所（額縁）」に、処理が終わった画像を表示する
        frame_placeholder.image(processed_image, channels="BGR")

# （ループが終わったら）使い終わったカメラを解放する（お片付け）
cap.release()
cv2.destroyAllWindows()

# もし今のモードが「Stop」なら、メッセージを表示する
if st.session_state['mode'] == 'Stop':
    st.success("処理を停止しました。")