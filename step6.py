# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np 

# MediaPipeという道具箱から、「絵を描く道具」と「顔の細かい特徴を見つける専門家」を準備
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


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

# 「顔の特徴」ボタン。押されると、「状態いれもの」に「FaceMesh」という文字を入れる
if st.sidebar.button("✨ 顔の特徴 (メッシュ)"):
    st.session_state['mode'] = 'FaceMesh'
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

# 「顔の細かい特徴を見つける専門家」を呼び出して、準備してもらう
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    # カメラが起動していて、かつ「停止」モードではない間、ずっと繰り返す
    while cap.isOpened() and st.session_state['mode'] != 'Stop':
        # カメラから1枚の画像(フレーム)を読み込む
        success, image = cap.read()
        if not success:
            break

        # 映像を鏡のように左右反転させる
        image = cv2.flip(image, 1)
        # 処理した後の画像を入れるための変数を用意
        processed_image = image

        # 画像の色を、AIが理解しやすい「RGB」形式に変換する
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # もし今のモードが「FaceMesh」なら、顔の特徴点検出を行う
        if st.session_state['mode'] == 'FaceMesh':
            # 「専門家」に画像を見せて、顔の細かい点（478個！）を探してもらう
            results = face_mesh.process(image_rgb)
            
            # もし顔の点が見つかったら、それらを線で結んで網（メッシュ）を描く
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=processed_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION, # 点と点を結ぶ線の情報
                        landmark_drawing_spec=None, # 点自体は描かない
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)) # 線の色や太さ

        # 準備しておいた「空の場所（額縁）」に、処理が終わった画像を表示する
        frame_placeholder.image(processed_image, channels="BGR")

# （ループが終わったら）使い終わったカメラを解放する（お片付け）
cap.release()
cv2.destroyAllWindows()