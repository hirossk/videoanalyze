# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2
import mediapipe as mp
from processors.emoji_drawer import draw_face_emoji

# MediaPipeという道具箱から、「絵を描く道具」と「顔を見つける専門家」を準備
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection


# --- ここからWebページの見た目を作る ---

# ページ全体の設定。layout="wide" で画面を広く使う（最初のStreamlit命令にするのがルール）
st.set_page_config(page_title="顔の検出アプリ", page_icon="🙂", layout="wide")

# Webページに一番大きな「看板（タイトル）」を出す
st.title("📹 リアルタイムAI解析アプリを作ろう！")
# st.caption() で、タイトルの下に小さな説明文を出す
st.caption("AI（MediaPipe）で顔を検出し、顔の位置に絵文字を表示します")


# --- ボタンが押されたときの「状態」を覚えておく仕組み ---

# もし「mode」という名前の「状態いれもの」がなければ、最初に作っておく
# 最初は「止まっている(Stop)」状態にしておく
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

# with st.sidebar: と書くと、この中の命令がぜんぶ左側のサイドバーに表示される
with st.sidebar:
    st.header("🎛️ コントロール")
    st.caption("ボタンでモードを切り替えます")

    # 「顔の検出」ボタン。押されると、「状態いれもの」に「Face」という文字を入れる
    # width="stretch" でボタンをサイドバーの幅いっぱいに広げる
    if st.button("🙂 顔の検出", width="stretch", type="primary"):
        st.session_state['mode'] = 'Face'

    # 「停止」ボタン。押されると、「状態いれもの」に「Stop」という文字を入れる
    if st.button("🛑 停止", width="stretch"):
        st.session_state['mode'] = 'Stop'

    # st.divider() で区切り線を引く
    st.divider()

    # st.badge() で、今の状態を色つきラベルでわかりやすく表示する
    if st.session_state['mode'] == 'Stop':
        st.badge("停止中", icon="⚪", color="grey")
    else:
        st.badge(f"実行中: {st.session_state['mode']}", icon="🟢", color="green")


# --- ここからカメラの映像を処理する ---

# 映像を画面幅の約80%に収めるため、中央のカラムに表示する（左右に余白を作る [1:8:1]）
_, _col_video, _ = st.columns([1, 8, 1])
frame_placeholder = _col_video.empty()
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
                # mp_drawing.draw_detection(processed_image, results.detections[0])
                # pass
                # もし複数の顔が見つかったら、全ての顔に対して四角を描く
                # for文は繰り返すという意味
                for detection in results.detections:
                    pass
                    # 四角とマーカーを描く
                    # mp_drawing.draw_detection(processed_image, detection)
                    # 顔の位置を取得して、絵文字を描く
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = processed_image.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    
                    face_emoji = ""  # 顔の絵文字
                    # # 顔の中心座標を計算
                    processed_image = draw_face_emoji(processed_image, x, y, w, h, face_emoji)

        # 準備しておいた「空の場所（額縁）」に、処理が終わった画像を表示する
        # width="stretch" で映像を画面の幅いっぱいに広げる
        frame_placeholder.image(processed_image, channels="BGR", width="stretch")

# （ループが終わったら）使い終わったカメラを解放する（お片付け）
cap.release()
cv2.destroyAllWindows()

# もし今のモードが「Stop」なら、メッセージを表示する
if st.session_state['mode'] == 'Stop':
    st.success("処理を停止しました。")