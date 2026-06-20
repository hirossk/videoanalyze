# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import requests # Webから画像をダウンロードするために追加

# ページ全体の設定。layout="wide" で画面を広く使う（最初のStreamlit命令にするのがルール）
st.set_page_config(page_title="バーチャル・サングラス", page_icon="😎", layout="wide")

# MediaPipeという道具箱から、「絵を描く道具」と「顔の細かい特徴を見つける専門家」を準備
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# サングラスの大きさを調整するための定数
SUNG_WIDTH_FACTOR = 1.5  # サングラスの幅を目の間の距離の何倍にするか

# @st.cache_data は、一度読み込んだ画像を賢く使いまわすための印
@st.cache_data
def load_image_from_url(url):
    """
    URLから画像をダウンロードして、OpenCVで使える形に変換する関数
    """
    try:
        response = requests.get(url)
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 3:
            alpha_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
            img = np.concatenate([img, alpha_channel], axis=2)
        return img
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        return None

# --- ここからWebページの見た目を作る ---
st.title("😎 バーチャル・サングラス アプリ")
st.caption("AI（MediaPipe FaceMesh）で顔のパーツを検出し、サングラス画像を顔に合成します")

# --- ボタンが押されたときの「状態」を覚えておく仕組み ---
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

# with st.sidebar: と書くと、この中の命令がぜんぶ左側のサイドバーに表示される
with st.sidebar:
    st.header("🎛️ コントロール")
    st.caption("ボタンでモードを切り替えます")

    # width="stretch" でボタンをサイドバーの幅いっぱいに広げる
    if st.button("AIカメラ起動", width="stretch", type="primary"):
        st.session_state['mode'] = 'Sunglasses'
    if st.button("🛑 停止", width="stretch"):
        st.session_state['mode'] = 'Stop'

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
cap = cv2.VideoCapture(0)

# サングラスの画像をローカルファイルから読み込む
sunglasses_img = cv2.imread("glasses/glasses1.png", cv2.IMREAD_UNCHANGED)


# 「顔の細かい特徴を見つける専門家」を呼び出して、準備してもらう
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened() and st.session_state['mode'] != 'Stop':
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        processed_image = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if st.session_state['mode'] == 'Sunglasses' and sunglasses_img is not None:
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    # --- [ステップ1] まずは顔の網目を表示して、AIが顔を見つけているか確認しよう ---
                    # 以下の5行のコメントを外すと、顔に緑色のメッシュが表示される
                    # mp_drawing.draw_landmarks(
                    #     image=processed_image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                    #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))


                    # --- [ステップ2] 次に、サングラスを置く「目」の場所を見つけて、印をつけてみよう ---
                    # 以下の8行のコメントを外すと、両目尻に赤い丸が表示される
                    landmarks = face_landmarks.landmark
                    # 左目尻(33番)と右目尻(263番)の座標を取得
                    left_eye = landmarks[33]
                    right_eye = landmarks[263]
                    
                    ih, iw, _ = image.shape
                    
                    left_eye_x, left_eye_y = int(left_eye.x * iw), int(left_eye.y * ih)
                    right_eye_x, right_eye_y = int(right_eye.x * iw), int(right_eye.y * ih)
                    
                    # cv2.circle(processed_image, (left_eye_x, left_eye_y), 5, (0, 0, 255), -1)
                    # cv2.circle(processed_image, (right_eye_x, right_eye_y), 5, (0, 0, 255), -1)


                    # --- [ステップ3] 最後に、計算した場所にサングラスの画像を合成しよう ---
                    # 以下のコメントを全て外すと、サングラスが表示される
                    
                    # # サングラスの幅を、両目の間の距離に合わせて決める
                    sunglasses_width = int(abs(right_eye_x - left_eye_x) * SUNG_WIDTH_FACTOR)
                    # # 元の画像の縦横比を保ったまま、高さを計算
                    sh, sw, _ = sunglasses_img.shape
                    sunglasses_height = int(sunglasses_width * (sh / sw))
                    
                    # # サングラスの大きさを変更
                    resized_sunglasses = cv2.resize(sunglasses_img, (sunglasses_width, sunglasses_height))
                    
                    # # --- カメラ映像にサングラスを合成 ---
                    # # サングラスを置く中心の座標を決める
                    center_x = (left_eye_x + right_eye_x) // 2
                    center_y = (left_eye_y + right_eye_y) // 2
                    
                    # # サングラスを置く左上の座標を計算
                    top_left_x = center_x - sunglasses_width // 2
                    top_left_y = center_y - sunglasses_height // 2

                    # # 元の画像から、サングラスを置く部分（ROI）を切り出す
                    # # エラーを防ぐため、画像の外にはみ出さないように座標を調整
                    if top_left_y < 0 or top_left_x < 0 or top_left_y + sunglasses_height > ih or top_left_x + sunglasses_width > iw:
                        continue # はみ出す場合はこのフレームの処理をスキップ
                    
                    roi = processed_image[top_left_y: top_left_y + sunglasses_height, top_left_x: top_left_x + sunglasses_width]

                    # # サングラス画像の透明な部分をマスクとして使う
                    mask = resized_sunglasses[:, :, 3]
                    mask_inv = cv2.bitwise_not(mask)
                    
                    # # マスクを使って、元の画像からサングラス部分をくり抜く
                    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    # # サングラス画像から、背景が透明なサングラス本体だけを取り出す
                    fg = cv2.bitwise_and(resized_sunglasses, resized_sunglasses, mask=mask)
                    
                    # # くり抜いた背景と、サングラス本体を合体させる
                    combined = cv2.add(bg, fg[:,:,:3])
                    
                    # # 最後に、元の画像に合成したサングラスを上書きする
                    # processed_image[top_left_y: top_left_y + sunglasses_height, top_left_x: top_left_x + sunglasses_width] = combined

        # 準備しておいた「空の場所（額縁）」に、処理が終わった画像を表示する
        # width="stretch" で映像を画面の幅いっぱいに広げる
        frame_placeholder.image(processed_image, channels="BGR", width="stretch")

# （ループが終わったら）使い終わったカメラを解放する（お片付け）
cap.release()
cv2.destroyAllWindows()
