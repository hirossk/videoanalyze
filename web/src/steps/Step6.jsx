import { useRef, useState } from "react";
import { useCameraLoop } from "../hooks/useCameraLoop.js";
import { Stage } from "../components/Stage.jsx";
import { Controls } from "../components/Controls.jsx";
import { createPoseLandmarker, DrawingUtils, PoseLandmarker } from "../lib/vision.js";

// 体のランドマーク番号（MediaPipe Pose）
const LEFT_EAR = 7;
const RIGHT_EAR = 8;
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;

// Step 6 ── ストレッチ回数カウンター
export default function Step6() {
  const [running, setRunning] = useState(false);
  const [counter, setCounter] = useState(0); // 画面に出す回数
  const [stageLabel, setStageLabel] = useState("水平");

  // 毎フレームで使い回す「状態のメモ帳」（再描画を避けるため ref に置く）
  const stageRef = useRef("水平");
  const prevRef = useRef("水平");
  const counterRef = useRef(0);

  const { videoRef, canvasRef, status, error } = useCameraLoop({
    running,
    setup: () => {
      // 起動のたびにカウンターをリセット
      stageRef.current = "水平";
      prevRef.current = "水平";
      counterRef.current = 0;
      setCounter(0);
      setStageLabel("水平");
      return createPoseLandmarker();
    },

    process: (ctx, video, pose) => {
      if (!pose) return;
      const drawing = new DrawingUtils(ctx);
      const result = pose.detectForVideo(ctx.canvas, performance.now());
      const landmarks = result.landmarks?.[0];

      let current = stageRef.current;

      if (landmarks) {
        // ★穴埋め：今は「耳(EAR)」の高さを見ています。
        //   これを「肩(SHOULDER)」に変えると、より正しくストレッチを判定できます。
        //   RIGHT_EAR → RIGHT_SHOULDER、LEFT_EAR → LEFT_SHOULDER に書き換えてみよう。
        const right = landmarks[RIGHT_EAR];
        const left = landmarks[LEFT_EAR];

        // 左右の「高さ(y座標)」の差を計算する
        const diff = right.y - left.y;

        // 「水平」と判断する“ゆるさ”。小さくすると判定が厳しくなる
        const threshold = 0.03;
        if (Math.abs(diff) < threshold) current = "水平";
        else if (diff > threshold) current = "左下がり";
        else if (diff < -threshold) current = "右下がり";

        // 「水平→左下がり→水平→右下がり→水平」と進んだら1回カウント
        if (prevRef.current !== current) {
          const stage = stageRef.current;
          if (stage === "水平" && current === "左下がり") stageRef.current = "左下がり";
          else if (stage === "左下がり" && current === "水平") stageRef.current = "水平2";
          else if (stage === "水平2" && current === "右下がり") stageRef.current = "右下がり";
          else if (stage === "右下がり" && current === "水平") {
            stageRef.current = "水平";
            // 1往復したのでカウント
            setCounter(counterRef.current);
          }
          setStageLabel(stageRef.current);
        }
        prevRef.current = current;

        // 体の関節を線で結んで描く
        drawing.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
          color: "#ffffff",
          lineWidth: 2,
        });
        drawing.drawLandmarks(landmarks, { color: "#ff4b6e", radius: 3 });
      }

      // カウンター表示用の背景と文字
      ctx.fillStyle = "rgba(16, 117, 245, 0.9)";
      ctx.fillRect(0, 0, 300, 73);
      ctx.fillStyle = "#000";
      ctx.font = "18px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("回数", 15, 26);
      ctx.fillStyle = "#fff";
      ctx.font = "bold 28px sans-serif";
      ctx.fillText(String(counterRef.current), 12, 58);
      ctx.fillStyle = "#000";
      ctx.font = "20px sans-serif";
      ctx.fillText(`状態: ${stageRef.current}`, 120, 45);
    },
  });

  return (
    <div>
      <div className="step-header">
        <h2>🤸 ストレッチ回数カウンター</h2>
        <p className="caption">AI（MediaPipe）で姿勢を検出し、腕のストレッチ回数を自動でカウントします</p>
      </div>

      <div className="layout">
        <div>
          <Controls
            running={running}
            onStart={() => setRunning(true)}
            onStop={() => setRunning(false)}
            startLabel="🤸‍♀️ ストレッチ開始"
          />
          <div className="controls" style={{ marginTop: 16 }}>
            <h3>🤸 ストレッチ回数</h3>
            <div style={{ fontSize: 40, fontWeight: 700 }}>{counter}</div>
            <span className="badge grey">状態: {stageLabel}</span>
          </div>
        </div>
        <div>
          <Stage videoRef={videoRef} canvasRef={canvasRef} status={status} error={error} />
          <div className="hint-box">
            <b>やること：</b> ①起動して上体を左右に倒す → ②カウントが増えるか確認 → ③
            <code>RIGHT_EAR / LEFT_EAR</code> を <code>RIGHT_SHOULDER / LEFT_SHOULDER</code>{" "}
            に変えて精度を比べてみよう。
          </div>
        </div>
      </div>
    </div>
  );
}
