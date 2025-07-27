import streamlit as st
import cv2
from processors import cartoon_styler

st.title("📹 カートゥーン風エフェクトデモ")
st.sidebar.markdown("### 解析モードを選択してください")

if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Stop'

# --- UI要素（ボタンとスライダー） ---
if st.sidebar.button("🎨 カートゥーン風エフェクト"):
    st.session_state['mode'] = 'Cartoon'
if st.sidebar.button("🛑 停止"):
    st.session_state['mode'] = 'Stop'

# 「Cartoon」モードの時だけスライダーを表示
if st.session_state['mode'] == 'Cartoon':
    st.sidebar.subheader("パラメータ調整")
    bilateral_d = st.sidebar.slider("色の滑らかさ (d)", 3, 15, 9, step=2)
    bilateral_sigmaColor = st.sidebar.slider("色の範囲 (sigmaColor)", 50, 500, 300, step=10)
    median_ksize = st.sidebar.slider("輪郭の滑らかさ (ksize)", 3, 15, 7, step=2)
    adaptive_blockSize = st.sidebar.slider("輪郭の細かさ (blockSize)", 3, 25, 13, step=2)
    adaptive_C = st.sidebar.slider("輪郭の強さ (C)", 0, 10, 2, step=1)

# --- カメラ処理 ---
frame_placeholder = st.empty()

if st.session_state['mode'] == 'Cartoon':
    # カメラを一度だけ起動
    cap = cv2.VideoCapture(0)

    # 停止ボタンが押されるまでループを続ける
    while st.session_state['mode'] == 'Cartoon':
        success, image = cap.read()
        if not success:
            st.error("カメラの読み込みに失敗しました。")
            break

        image = cv2.flip(image, 1)
        
        # スライダーの値を使って画像を加工
        processed_image = cartoon_styler.process(
            image,
            bilateral_d=bilateral_d,
            bilateral_sigmaColor=bilateral_sigmaColor,
            bilateral_sigmaSpace=bilateral_sigmaColor, # sigmaSpaceはColorと同じで良いことが多い
            median_ksize=median_ksize,
            adaptive_blockSize=adaptive_blockSize,
            adaptive_C=adaptive_C
        )
        
        frame_placeholder.image(processed_image, channels="BGR")
    
    # ループが終了したら、カメラを解放
    cap.release()

else:
    frame_placeholder.write("「カートゥーン風エフェクト」ボタンを押してください。")