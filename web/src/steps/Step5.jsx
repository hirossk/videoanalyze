import { useEffect, useRef, useState } from "react";
import { useCameraLoop } from "../hooks/useCameraLoop.js";
import { Stage } from "../components/Stage.jsx";
import { Controls } from "../components/Controls.jsx";
import { createFaceLandmarker, DrawingUtils, FaceLandmarker } from "../lib/vision.js";

// サングラスの幅を、両目の間の距離の何倍にするか
const SUNG_WIDTH_FACTOR = 1.5;

// Step 5 ── バーチャル・サングラスをかけよう
export default function Step5() {
  const [running, setRunning] = useState(false);

  // サングラス画像を読み込んでおく（public/glasses/glasses1.png）
  const glassesRef = useRef(null);
  useEffect(() => {
    const img = new Image();
    img.src = "/glasses/glasses1.png";
    img.onload = () => (glassesRef.current = img);
  }, []);

  const { videoRef, canvasRef, status, error } = useCameraLoop({
    running,
    setup: () => createFaceLandmarker(),

    process: (ctx, video, faceMesh) => {
      if (!faceMesh) return;
      const drawing = new DrawingUtils(ctx);
      const W = ctx.canvas.width;
      const H = ctx.canvas.height;

      const result = faceMesh.detectForVideo(ctx.canvas, performance.now());

      for (const landmarks of result.faceLandmarks ?? []) {
        // ★ステップ1（穴埋め①）：顔のメッシュを表示する → 下のコメントを外す
        // drawing.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#00ff0055", lineWidth: 1 });

        // 左目尻(33番)と右目尻(263番)の座標を取得（0〜1の割合をピクセルに直す）
        const leftEye = landmarks[33];
        const rightEye = landmarks[263];
        const leftEyeX = leftEye.x * W;
        const leftEyeY = leftEye.y * H;
        const rightEyeX = rightEye.x * W;
        const rightEyeY = rightEye.y * H;

        // ★ステップ2（穴埋め②）：両目の位置に赤い丸を表示 → 下のコメントを外す
        // ctx.fillStyle = "red";
        // ctx.beginPath(); ctx.arc(leftEyeX, leftEyeY, 5, 0, Math.PI * 2); ctx.fill();
        // ctx.beginPath(); ctx.arc(rightEyeX, rightEyeY, 5, 0, Math.PI * 2); ctx.fill();

        // ★ステップ3（穴埋め③）：サングラスを合成する → 下のブロックのコメントを外す
        const glasses = glassesRef.current;
        if (glasses) {
          // サングラスの幅を、両目の間の距離に合わせて決める
          const sgWidth = Math.abs(rightEyeX - leftEyeX) * SUNG_WIDTH_FACTOR;
          // 元画像の縦横比を保ったまま高さを計算
          const sgHeight = sgWidth * (glasses.height / glasses.width);
          // サングラスを置く中心と左上の座標
          const centerX = (leftEyeX + rightEyeX) / 2;
          const centerY = (leftEyeY + rightEyeY) / 2;
          const topLeftX = centerX - sgWidth / 2;
          const topLeftY = centerY - sgHeight / 2;

          // drawImage は透明部分(PNGのアルファ)をそのまま活かして合成してくれる
          ctx.drawImage(glasses, topLeftX, topLeftY, sgWidth, sgHeight);
        }
      }
    },
  });

  return (
    <div>
      <div className="step-header">
        <h2>😎 バーチャル・サングラス アプリ</h2>
        <p className="caption">AI（FaceMesh）で顔のパーツを検出し、サングラス画像を顔に合成します</p>
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
            <b>やること：</b> ①メッシュ表示の行を外す → ②両目に赤い丸の行を外す → ③
            <code>ctx.drawImage(glasses, ...)</code> を外してサングラス合成。
          </div>
        </div>
      </div>
    </div>
  );
}
