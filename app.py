# -*- coding: utf-8 -*-
"""
🎓 AI画像認識 体験デモ（オープンキャンパス用）- マルチページアプリケーション
最新のStreamlit機能を活用した改善版
"""

import streamlit as st

st.set_page_config(
    page_title="AI体験デモ | オープンキャンパス",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 AI画像認識 体験デモ")
st.subheader("オープンキャンパスへようこそ！")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### 🎯 このデモについて

    AIはあなたの**顔**や**手**を認識して、
    自動的に飾りをつけたり、
    エフェクトを表示したりします！

    #### ✨ できること
    - 😎 顔を見つけるとサングラスや猫耳がつく
    - ✌️ 手のサインを認識してエフェクト表示
    - 🙌 ポーズも判定できる

    """)

with col2:
    st.info("""
    ### 🚀 さあ、始めよう！

    右の「ライブデモを開始」ボタンを押して、
    あなたの顔と手を認識させてみてください。

    わからないことがあれば、「ジェスチャーガイド」を
    チェック！詳しく説明しています。
    """)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ ライブデモを開始", use_container_width=True, key="demo_button"):
        st.switch_page("pages/01_live_demo.py")

with col2:
    if st.button("✌️ ジェスチャーガイド", use_container_width=True, key="guide_button"):
        st.switch_page("pages/02_gesture_guide.py")

with col3:
    if st.button("ℹ️ 詳しく知る", use_container_width=True, key="about_button"):
        st.switch_page("pages/03_about.py")

st.divider()

with st.expander("🔧 トラブルシューティング"):
    st.markdown("""
    **カメラが起動しない場合**
    - 他のアプリがカメラを使っていないか確認してください
    - ブラウザの設定でカメラへのアクセスを許可してください

    **動作が遅い場合**
    - デモページの「設定」で画像サイズを小さくしてみてください
    - バックグラウンドアプリを終了してください

    **ジェスチャーが認識されない場合**
    - 「ジェスチャーガイド」を読んでみてください
    - 照明を改善してみてください
    - カメラに近づいてみてください
    """)
