import cv2
import mediapipe as mp
import numpy as np
# Pillowライブラリをインポート
from PIL import ImageFont, ImageDraw, Image

# MediaPipeから「姿勢を見つける専門家」と「絵を描く道具」を準備
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def put_japanese_text(image, text, point, font_path, font_size, color):
    """
    Pillowを使って画像に日本語を描画する関数
    """
    # OpenCVの画像(numpy.ndarray)をPillowの画像に変換
    img_pil = Image.fromarray(image)
    # 描画用のオブジェクトを作成
    draw = ImageDraw.Draw(img_pil)
    # 指定したフォントとサイズで、フォントオブジェクトを作成
    font = ImageFont.truetype(font_path, font_size)
    # 指定した位置にテキストを描画
    draw.text(point, text, font=font, fill=color)
    # Pillowの画像をOpenCVの画像形式に変換して返す
    return np.array(img_pil)


# 「店長(app.py)」から呼び出される、ストレッチ判定の専門処理
def process(image, pose, counter, stage, prev_current):
    # AIが理解しやすいように、画像の色を変換する
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 「姿勢の専門家」に画像を見せて、結果をもらう
    results = pose.process(image_rgb)
    
    # 今の体の傾きを入れる「いれもの」。最初は前の状態を維持する
    current = stage
    
    # try...exceptは、エラーが起きてもプログラムが止まらないようにするおまじない
    # (例：カメラに人が映っていないときなど)
    try:
        # 結果から、体中の関節の座標リストを取り出す
        landmarks = results.pose_landmarks.landmark
        
        # --- 肩の傾きを計算するエリア ---
        
        # 座標リストから「右肩」と「左肩」の場所(x, y座標)を取り出す
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        
        # 右肩と左肩の「高さ(y座標)」の差を計算する
        diff = right_shoulder[1] - left_shoulder[1]
        
        # ★★★ ここが一番大事な調整ポイント！ ★★★
        # 「水平」と判断する「ゆるさ」の度合い。
        # 小さくすると、少しの傾きでも「右下がり」「左下がり」と判断される（判定が厳しくなる）
        # 大きくすると、多少傾いていても「水平」とみなされる（判定が甘くなる）
        threshold = 0.03

        # 高さの差(diff)が、調整したゆるさ(threshold)の範囲内だったら「水平」と判断
        if abs(diff) < threshold:
            current = "水平"
        # もし差がプラスなら、右肩が下がっているので「右下がり」
        elif diff > threshold:
            current = "左下がり"
        # もし差がマイナスなら、左肩が下がっているので「左下がり」
        elif diff < -threshold:
            current = "右下がり"

        # --- ストレッチの段階を進めるエリア ---
        
        # もし１つ前の傾き(prev_current)と今の傾き(current)が変化した瞬間だったら…
        if prev_current != current:
            # 「水平」→「右下がり」→「水平」→「左下がり」→「水平」の順で進んだら1回とカウントする
            if stage == "水平" and current == "左下がり":
                stage = "左下がり"
            elif stage == "左下がり" and current == "水平":
                stage = "水平2"
            elif stage == "水平2" and current == "右下がり":
                stage = "右下がり"
            elif stage == "右下がり" and current == "水平":
                counter += 1
                stage = "水平"
        
        # 今の傾きを「１つ前の傾き」として覚えておく
        prev_current = current

    except:
        # 人が映っていないなどでエラーになった場合は、何もしない
        pass

    # --- 画面に文字や絵を描くエリア ---
    
    # カウンター表示用の四角い背景を描く
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
    
    # 画面への描画処理
    font_path = "fonts/NotoSansJP-Regular.ttf" # フォントファイルを指定

    cv2.rectangle(image, (0,0), (300,73), (245,117,16), -1)
    
    # put_japanese_text関数を使って日本語を描画
    image = put_japanese_text(image, "回数", (15,10), font_path, 20, (0,0,0))
    image = put_japanese_text(image, str(counter), (10,30), font_path, 40, (255,255,255))
    image = put_japanese_text(image, f"状態: {stage}", (130,35), font_path, 25, (0,0,0))


    # 見つけ出した体の関節（ポーズ）を線で結んで描く
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 処理が終わった画像と、更新したカウンターやステージの情報を「店長」に返す
    return image, counter, stage, prev_current