# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2
import mediapipe as mp

# MediaPipeという道具箱から、「絵を描く道具」と「顔を見つける専門家」を準備
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection


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

# 「顔の検出」ボタン。押されると、「状態いれもの」に「Face」という文字を入れる
if st.sidebar.button("🙂 顔の検出"):
    st.session_state['mode'] = 'Face'

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

# 「顔を見つける専門家」を呼び出して、準備してもらう
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    
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

        # もし今のモードが「Face」なら、顔の検出を行う
        if st.session_state['mode'] == 'Face':
            # 「顔を見つける専門家」に画像を見せて、顔を探してもらう
            results = face_detection.process(image_rgb)
            
            # もし顔が見つかったら、その場所に四角を描く
            if results.detections:
                # 最初に見つかった顔に対して、四角を描く
                mp_drawing.draw_detection(processed_image, results.detections[0])
                pass
                # for detection in results.detections:
                    # mp_drawing.draw_detection(processed_image, detection)

        # 準備しておいた「空の場所（額縁）」に、処理が終わった画像を表示する
        frame_placeholder.image(processed_image, channels="BGR")

# （ループが終わったら）使い終わったカメラを解放する（お片付け）
cap.release()
cv2.destroyAllWindows()

# もし今のモードが「Stop」なら、メッセージを表示する
if st.session_state['mode'] == 'Stop':
    st.success("処理を停止しました。")