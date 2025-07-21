import cv2
import mediapipe as mp

# MediaPipeの描画ユーティリティと顔検出ソリューションを準備
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

# Webカメラから入力を開始
# 引数0はPCに内蔵されたデフォルトのカメラを指します
cap = cv2.VideoCapture(0)

# 顔検出モデルを読み込む
# min_detection_confidence: 顔として検出するための最小信頼度（0.0〜1.0）
# 0.5は50%以上の確信がある場合に顔として認識するという意味
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:

    # カメラが起動している間、ループを続ける
    while cap.isOpened():
        # カメラから1フレーム（1枚の画像）を読み込む
        success, image = cap.read()
        if not success:
            print("カメラフレームの読み込みに失敗しました。")
            continue

        # パフォーマンス向上のため、画像を書き込み不可として参照渡しする
        image.flags.writeable = False
        # MediaPipeで処理するために、画像をBGR形式からRGB形式に変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipeの顔検出を実行
        results = face_detection.process(image)

        # 処理のために、画像を書き込み可能に戻し、RGBからBGRに再変換
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 検出結果（results.detections）が存在する場合、顔に枠を描画
        if results.detections:
            for detection in results.detections:
                # mp_drawingユーティリティを使って、バウンディングボックス（四角い枠）を描画
                mp_drawing.draw_detection(image, detection)
        
        # 結果を画面に表示
        cv2.imshow('MediaPipe Face Detection', image)
        
        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 使い終わったら、カメラとウィンドウを解放する
cap.release()
cv2.destroyAllWindows()