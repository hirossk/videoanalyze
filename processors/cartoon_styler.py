import cv2
import numpy as np

def process(image):
    """
    入力された画像をカートゥーン風に加工する
    """
    # 1. 色を滑らかにする（油絵風）
    # cv2.bilateralFilterは、輪郭を保持したままノイズを削減するのに優れている
    img_color = cv2.bilateralFilter(image, d=9, sigmaColor=300, sigmaSpace=300)

    # 2. 輪郭線を抽出する（マンガ風）
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7) # ノイズ削減
    # adaptiveThresholdで、画像の各領域で最適な閾値を計算し、クッキリした線画を生成
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)

    # 3. 色と輪郭線を合成
    # 輪郭線は白黒なので、カラー画像と合成するために3チャンネルに変換
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    
    # cv2.bitwise_andで、輪郭線の黒い部分だけをカラー画像に重ねる
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    
    return img_cartoon