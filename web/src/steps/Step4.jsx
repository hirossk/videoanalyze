import { useState } from "react";
import { useCameraLoop } from "../hooks/useCameraLoop.js";
import { Stage } from "../components/Stage.jsx";
import { Controls } from "../components/Controls.jsx";
import { createHandLandmarker, DrawingUtils, HandLandmarker } from "../lib/vision.js";

// Step 4 ── 手を検出して指に名前をつけよう
export default function Step4() {
  const [running, setRunning] = useState(false);

  const { videoRef, canvasRef, status, error } = useCameraLoop({
    running,
    // 最初に1回だけ「手を見つける専門家」を準備する
    setup: () => createHandLandmarker(2),

    process: (ctx, video, hands) => {
      if (!hands) return;
      const drawing = new DrawingUtils(ctx);
      const W = ctx.canvas.width;
      const H = ctx.canvas.height;

      // 「手を見つける専門家」に画像を見せて、手を探してもらう（先に検出する）
      const result = hands.detectForVideo(ctx.canvas, performance.now());

      // ★穴埋め⑤：背景を真っ暗にして骨格だけ表示 → 下のコメントを外す
      // ctx.fillStyle = "black";
      // ctx.fillRect(0, 0, W, H);

      // もし手が見つかったら、そのぶんだけ繰り返す
      for (const landmarks of result.landmarks ?? []) {
        // ★穴埋め②：手の骨格を描く → 下の2行のコメントを外す
        // drawing.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
        // drawing.drawLandmarks(landmarks, { color: "#FF0000", radius: 3 });

        // ★穴埋め③：指の名前。"" を「親指」「人差し指」…に書き換えよう
        const fingerNames = ["", "", "", "", ""];
        const fingerTips = [4, 8, 12, 16, 20]; // 各指先のランドマーク番号

        // ★穴埋め④：各指先に名前を表示する → 下のブロックのコメントを外す
        // for (let i = 0; i < fingerTips.length; i++) {
        //   const tip = landmarks[fingerTips[i]];
        //   ctx.fillStyle = "#00ff00";
        //   ctx.font = "20px sans-serif";
        //   ctx.textAlign = "center";
        //   ctx.fillText(fingerNames[i], tip.x * W, tip.y * H - 10);
        // }
      }
    },
  });

  return (
    <div>
      <div className="step-header">
        <h2>📹 リアルタイムAI解析アプリ ── 手の検出</h2>
        <p className="caption">AI（MediaPipe）で手の骨格を検出し、各指先にラベルを表示します</p>
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
            <b>やること：</b> ①手をかざす → ②骨格を描く2行を外す → ③
            <code>fingerNames</code> を指の名前に → ④名前表示の <code>for</code>{" "}
            ブロックを外す → ⑤背景を黒にして骨格だけ表示。
          </div>
        </div>
      </div>
    </div>
  );
}
