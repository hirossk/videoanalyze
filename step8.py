# 必要な「魔法の道具箱」を使えるように準備するおまじない
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import requests # Webから画像をダウンロードするために追加

# MediaPipeという道具箱から、「顔の細かい特徴を見つける専門家」を準備
mp_face_mesh = mp.solutions.face_mesh

# サングラスの大きさを調整するための定数
SUNG_WIDTH_FACTOR = 1.5  # サングラスの幅を目の間の距離の何倍にするか
# --- サングラスの画像をインターネットから読み込む準備 ---

# @st.cache_data は、一度読み込んだ画像を賢く使いまわすための印
@st.cache_data
def load_image_from_url(url):
    """
    URLから画像をダウンロードして、OpenCVで使える形に変換する関数
    """
    try:
        response = requests.get(url)
        # ダウンロードした画像データを数値の配列に変換
        img_array = np.frombuffer(response.content, np.uint8)
        # 数値の配列を画像データにデコード（変換）
        # cv2.IMREAD_UNCHANGEDは、透明な部分も読み込むための設定
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        # 画像が3チャンネル(RGB)の場合はアルファチャンネルを追加
        if img is not None and img.shape[2] == 3:
            # アルファ値255（完全不透明）を追加
            alpha_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
            img = np.concatenate([img, alpha_channel], axis=2)
        return img
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        return None

# --- ここからWebページの見た目を作る ---
st.title("😎 バーチャル・サングラス アプリ")
st.sidebar.markdown("モードを選択してください")

# --- ボタンが押されたときの「状態」を覚えておく仕組み ---
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

if st.sidebar.button("😎 サングラスをかける"):
    st.session_state['mode'] = 'Sunglasses'
if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'

st.sidebar.markdown(f"**現在のモード:** `{st.session_state['mode']}`")

# --- ここからカメラの映像を処理する ---
frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)

# サングラスの画像をダウンロード（最初に1回だけ実行される）
sunglasses_url = "https://irasutoya.jp/wp-content/uploads/2020/08/anim-shonsangurasu-no-irasuto-png-t-ka.png"
sunglasses_img = load_image_from_url(sunglasses_url)


# 「顔の細かい特徴を見つける専門家」を呼び出して、準備してもらう
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened() and st.session_state['mode'] != 'Stop':
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        # 処理後の画像を入れるための変数を用意
        processed_image = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # もし今のモードが「Sunglasses」なら、サングラスをかける処理を行う
        if st.session_state['mode'] == 'Sunglasses' and sunglasses_img is not None:
            results = face_mesh.process(image_rgb)
            
            # もし顔の点が見つかったら…
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    # --- サングラスの位置と大きさを計算 ---
                    landmarks = face_landmarks.landmark
                    # 左目尻(33)と右目尻(263)の座標を取得
                    left_eye = landmarks[33]
                    right_eye = landmarks[263]
                    
                    # 画像の幅と高さを取得
                    ih, iw, _ = image.shape
                    
                    # 座標をピクセル単位に変換
                    left_eye_x, left_eye_y = int(left_eye.x * iw), int(left_eye.y * ih)
                    right_eye_x, right_eye_y = int(right_eye.x * iw), int(right_eye.y * ih)
                    
                    # サングラスの幅を、両目の間の距離に合わせて決める
                    sunglasses_width = int(abs(right_eye_x - left_eye_x) * SUNG_WIDTH_FACTOR)
                    # 元の画像の縦横比を保ったまま、高さを計算
                    sh, sw, _ = sunglasses_img.shape
                    sunglasses_height = int(sunglasses_width * (sh / sw))
                    
                    # サングラスの大きさを変更
                    resized_sunglasses = cv2.resize(sunglasses_img, (sunglasses_width, sunglasses_height))
                    
                    # --- カメラ映像にサングラスを合成 ---
                    # サングラスを置く中心の座標を決める
                    center_x = (left_eye_x + right_eye_x) // 2
                    center_y = (left_eye_y + right_eye_y) // 2
                    
                    # サングラスを置く左上の座標を計算
                    top_left_x = center_x - sunglasses_width // 2
                    top_left_y = center_y - sunglasses_height // 2

                    # 元の画像から、サングラスを置く部分（ROI）を切り出す
                    roi = processed_image[top_left_y: top_left_y + sunglasses_height, top_left_x: top_left_x + sunglasses_width]

                    # サングラス画像の透明な部分をマスクとして使う
                    mask = resized_sunglasses[:, :, 3]
                    mask_inv = cv2.bitwise_not(mask)
                    
                    # マスクを使って、元の画像からサングラス部分をくり抜く
                    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    # サングラス画像から、背景が透明なサングラス本体だけを取り出す
                    fg = cv2.bitwise_and(resized_sunglasses, resized_sunglasses, mask=mask)
                    
                    # くり抜いた背景と、サングラス本体を合体させる
                    combined = cv2.add(bg, fg[:,:,:3])
                    
                    # 最後に、元の画像に合成したサングラスを上書きする
                    processed_image[top_left_y: top_left_y + sunglasses_height, top_left_x: top_left_x + sunglasses_width] = combined

        # 準備しておいた「空の場所（額縁）」に、処理が終わった画像を表示する
        frame_placeholder.image(processed_image, channels="BGR")

# （ループが終わったら）使い終わったカメラを解放する（お片付け）
cap.release()
cv2.destroyAllWindows()
