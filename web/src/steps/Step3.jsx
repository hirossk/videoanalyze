import { useState } from "react";
import { useCameraLoop } from "../hooks/useCameraLoop.js";
import { Stage } from "../components/Stage.jsx";
import { Controls } from "../components/Controls.jsx";
import { createFaceDetector } from "../lib/vision.js";

// Step 3 ── 顔を検出しよう
// AI（MediaPipe）で顔を検出し、顔の位置に絵文字を表示します。
export default function Step3() {
  const [running, setRunning] = useState(false);

  const { videoRef, canvasRef, status, error } = useCameraLoop({
    running,
    // 最初に1回だけ「顔を見つける専門家」を準備する
    setup: () => createFaceDetector(),

    process: (ctx, video, faceDetector) => {
      if (!faceDetector) return;

      // 「顔を見つける専門家」に画像を見せて、顔を探してもらう
      const result = faceDetector.detectForVideo(ctx.canvas, performance.now());

      // もし顔が見つかったら、そのぶんだけ繰り返す
      for (const detection of result.detections) {
        // 顔の四角い範囲（ピクセル座標）
        const box = detection.boundingBox;
        const x = box.originX;
        const y = box.originY;
        const w = box.width;
        const h = box.height;

        // ★穴埋め②：顔に四角い枠を描く → 下の2行のコメントを外す
        // ctx.strokeStyle = "#00ff88";
        // ctx.lineWidth = 3;
        // ctx.strokeRect(x, y, w, h);

        // ★穴埋め③：好きな絵文字を入れよう（例: "😊" や "🤖"）
        const faceEmoji = "";
        if (faceEmoji) {
          ctx.font = `${Math.floor(h * 0.9)}px serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(faceEmoji, x + w / 2, y + h / 2);
        }
      }
    },
  });

  return (
    <div>
      <div className="step-header">
        <h2>📹 リアルタイムAI解析アプリ ── 顔の検出</h2>
        <p className="caption">AI（MediaPipe）で顔を検出し、顔の位置に絵文字を表示します</p>
      </div>

      <div className="layout">
        <Controls
          running={running}
          onStart={() => setRunning(true)}
          onStop={() => setRunning(false)}
        />
        <div>
          <Stage videoRef={videoRef} canvasRef={canvasRef} status={status} error={error} />
          <div className="hint-box">
            <b>やること：</b> ①起動して顔をカメラに向ける → ②枠を描く2行のコメントを外す
            → ③<code>faceEmoji</code> に好きな絵文字を入れる。
          </div>
        </div>
      </div>
    </div>
  );
}
