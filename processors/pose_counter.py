import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process(image, pose, counter, stage, prev_current):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    current = stage  # デフォルトは前回のstage
    try:

        landmarks = results.pose_landmarks.landmark
        # 右肩と左肩の座標を取得
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        
        # y座標の差で傾きを判定
        diff = right_shoulder[1] - left_shoulder[1]
        threshold = 0.03  # この値は調整してください

        if abs(diff) < threshold:
            current = "水平"
        elif diff > threshold:
            current = "右下がり"
        elif diff < -threshold:
            current = "左下がり"

        # 傾きが変化した瞬間だけ状態遷移
        if prev_current != current:
            # カウントロジック
            if stage == "水平" and current == "右下がり":
                stage = "右下がり"
            elif stage == "右下がり" and current == "水平":
                stage = "水平2"
            elif stage == "水平2" and current == "左下がり":
                stage = "左下がり"
            elif stage == "左下がり" and current == "水平":
                counter += 1
                stage = "水平"
        prev_current = current


    except:
        pass

    # 画面への描画処理
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
    cv2.putText(image, 'REPS', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, f'STAGE: {stage}', (100,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image, counter, stage, prev_current