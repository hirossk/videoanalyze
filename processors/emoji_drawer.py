from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

def draw_face_emoji(processed_image, x, y, w, h, face_emoji="😊", font_path="C:/Windows/Fonts/seguiemj.ttf"):
    """
    顔枠の中心に絵文字を描画し、OpenCV画像を返す
    """
    # 顔の中心座標
    center_x = x + w // 2
    center_y = y + h // 2

    # OpenCV画像をPillow画像に変換
    img_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 顔の大きさに合わせてフォントサイズを調整
    font_size = int(h * 0.8)
    font = ImageFont.truetype(font_path, font_size)

    # 顔文字のサイズを取得
    bbox = font.getbbox(face_emoji)
    emoji_w = bbox[2] - bbox[0]
    emoji_h = bbox[3] - bbox[1]
    # 顔枠の中心に顔文字の中心が来るように座標を調整
    draw.text(
        (center_x - emoji_w // 2, center_y - emoji_h // 2),
        face_emoji,
        font=font,
        fill=(0, 128, 255)
    )

    # Pillow画像をOpenCV画像に戻す
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)