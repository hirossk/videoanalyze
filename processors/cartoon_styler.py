import cv2
import numpy as np

def process(
    image,
    bilateral_d=9,
    bilateral_sigmaColor=300,
    bilateral_sigmaSpace=300,
    median_ksize=7,
    adaptive_blockSize=9,
    adaptive_C=2
):
    """
    入力された画像をカートゥーン風に加工する
    各種パラメータを指定可能
    """
    # 1. 色を滑らかにする（油絵風）
    img_color = cv2.bilateralFilter(
        image, d=bilateral_d, sigmaColor=bilateral_sigmaColor, sigmaSpace=bilateral_sigmaSpace
    )

    # 2. 輪郭線を抽出する（マンガ風）
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, median_ksize)
    img_edge = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=adaptive_blockSize,
        C=adaptive_C
    )

    # 3. 色と輪郭線を合成
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    
    return img_cartoon