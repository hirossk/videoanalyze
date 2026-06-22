import { useState } from "react";
import { useCameraLoop } from "../hooks/useCameraLoop.js";
import { Stage } from "../components/Stage.jsx";
import { Controls } from "../components/Controls.jsx";

// 画像をモノクロ（白黒）にする関数（Python の cv2.cvtColor(..., COLOR_BGR2GRAY) 相当）
function toGray(ctx) {
  const { width, height } = ctx.canvas;
  const imageData = ctx.getImageData(0, 0, width, height);
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    d[i] = d[i + 1] = d[i + 2] = gray;
  }
  ctx.putImageData(imageData, 0, 0);
}

// 画像を上下反転する関数（Python の cv2.flip(frame, 0) 相当）
function flipVertical(ctx) {
  const { width, height } = ctx.canvas;
  const tmp = document.createElement("canvas");
  tmp.width = width;
  tmp.height = height;
  tmp.getContext("2d").drawImage(ctx.canvas, 0, 0);
  ctx.save();
  ctx.clearRect(0, 0, width, height);
  ctx.translate(0, height);
  ctx.scale(1, -1);
  ctx.drawImage(tmp, 0, 0);
  ctx.restore();
}

// Step 2 ── カメラを表示しよう
export default function Step2() {
  const [running, setRunning] = useState(false);

  const { videoRef, canvasRef, status, error } = useCameraLoop({
    running,
    // このステップではAIは使わないので setup は無し
    process: (ctx) => {
      // 映像はすでにキャンバスに描かれている。ここで加工する。

      // ★穴埋め②：映像をモノクロにする → 下のコメントを外す
      // toGray(ctx);

      // ★穴埋め③：映像を上下反転する → 下のコメントを外す
      // flipVertical(ctx);
    },
  });

  return (
    <div>
      <div className="step-header">
        <h2>📷 シンプルカメラアプリ</h2>
        <p className="caption">PCのカメラ映像をブラウザにリアルタイムで表示します</p>
      </div>

      <div className="layout">
        <Controls
          running={running}
          onStart={() => setRunning(true)}
          onStop={() => setRunning(false)}
          startLabel="▶️ カメラ起動"
        />
        <div>
          <Stage
            videoRef={videoRef}
            canvasRef={canvasRef}
            status={status}
            error={error}
          />
          <div className="hint-box">
            <b>やること：</b> ①そのまま起動して映像を確認 → ②<code>toGray(ctx)</code>{" "}
            のコメントを外して白黒に → ③<code>flipVertical(ctx)</code>{" "}
            のコメントを外して上下反転。
          </div>
        </div>
      </div>
    </div>
  );
}
