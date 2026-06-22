import { useRef, useState } from "react";
import { useCameraLoop } from "../hooks/useCameraLoop.js";
import { Stage } from "../components/Stage.jsx";
import { createHandLandmarker, DrawingUtils, HandLandmarker } from "../lib/vision.js";
import { classifyHandGesture, decideWinner } from "../lib/jankenLogic.js";

const ROUND_SECONDS = 5; // カウントダウンの秒数
const SAMPLE_FRAMES = 5; // 直近何フレーム分から手を決めるか

// AI（コンピュータ）はランダムに手を選ぶ
function aiChoose() {
  const hands = ["GU", "PA", "CHOKI"];
  return hands[Math.floor(Math.random() * hands.length)];
}

// 配列の中で一番多く出てくるものを返す
function mostCommon(arr) {
  if (!arr.length) return null;
  const counts = {};
  let best = null;
  let bestN = 0;
  for (const x of arr) {
    counts[x] = (counts[x] || 0) + 1;
    if (counts[x] > bestN) {
      bestN = counts[x];
      best = x;
    }
  }
  return best;
}

// GU/CHOKI/PA を絵文字にして見やすく
const EMOJI = { GU: "✊", CHOKI: "✌️", PA: "🖐️" };
function toEmoji(h) {
  return EMOJI[h] || "-";
}

// キャンバスに背景つきの文字を描くヘルパー
function drawText(ctx, text, x, y, { size = 28, color = "#00ffff" } = {}) {
  ctx.font = `bold ${size}px sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  const w = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(x - 8, y - size, w + 16, size + 14);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

// 応用 ── じゃんけんAI
export default function Janken() {
  const [running, setRunning] = useState(false);
  const [scoreP, setScoreP] = useState(0);
  const [scoreA, setScoreA] = useState(0);
  const [lastResult, setLastResult] = useState("-");

  // ゲームの進行状況を覚えておくメモ帳
  const game = useRef({
    phase: "idle", // idle / countdown / result
    countdownEnd: null,
    buff: [],
    player: null,
    ai: null,
    resultUntil: 0,
  });

  function startRound() {
    game.current = {
      phase: "countdown",
      countdownEnd: null, // カメラ準備後、最初のフレームで決める
      buff: [],
      player: null,
      ai: null,
      resultUntil: 0,
    };
    setRunning(true);
  }

  function resetScore() {
    setScoreP(0);
    setScoreA(0);
    setLastResult("-");
  }

  const { videoRef, canvasRef, status, error } = useCameraLoop({
    running,
    setup: () => createHandLandmarker(1),

    process: (ctx, video, hands) => {
      if (!hands) return;
      const drawing = new DrawingUtils(ctx);
      const now = performance.now() / 1000;
      const g = game.current;

      // 手を検出して骨格を描く
      const result = hands.detectForVideo(ctx.canvas, performance.now());
      const landmarks = result.landmarks?.[0];
      if (landmarks) {
        drawing.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
          color: "#00ff00",
          lineWidth: 2,
        });
        drawing.drawLandmarks(landmarks, { color: "#0080ff", radius: 2 });
        // カウントダウン中だけ、手の形を記録する
        if (g.phase === "countdown") {
          const gesture = classifyHandGesture(landmarks);
          if (gesture !== "UNKNOWN") g.buff.push(gesture);
        }
      }

      // --- 勝敗表示中 ---
      if (g.phase === "result") {
        drawText(
          ctx,
          `YOU: ${toEmoji(g.player)}  AI: ${toEmoji(g.ai)}  => ${lastResult}`,
          20,
          60,
          { size: 30 }
        );
        if (now > g.resultUntil) {
          g.phase = "idle";
          setRunning(false); // カメラを止める
        }
        return;
      }

      // --- カウントダウン中 ---
      if (g.phase === "countdown") {
        if (g.countdownEnd === null) g.countdownEnd = now + ROUND_SECONDS;
        const secLeft = Math.ceil(g.countdownEnd - now);

        if (secLeft >= 1) {
          drawText(ctx, String(secLeft), 30, 130, { size: 96, color: "#ffff00" });
          return;
        }

        // --- 時間切れ → 判定 ---
        const recent = g.buff.slice(-SAMPLE_FRAMES);
        const player = mostCommon(recent);
        const ai = aiChoose();

        let res;
        if (!player) {
          res = "NO HAND";
        } else {
          res = decideWinner(player, ai);
          if (res === "WIN") setScoreP((s) => s + 1);
          else if (res === "LOSE") setScoreA((s) => s + 1);
        }

        g.player = player;
        g.ai = ai;
        g.phase = "result";
        g.resultUntil = now + 2.5;
        setLastResult(res);
        drawText(ctx, `YOU: ${toEmoji(player)}  AI: ${toEmoji(ai)}  => ${res}`, 20, 60, {
          size: 30,
        });
      }
    },
  });

  return (
    <div>
      <div className="step-header">
        <h2>✊✌️🖐️ じゃんけんAI</h2>
        <p className="caption">
          カメラに手を向けてグー・チョキ・パー！ {ROUND_SECONDS}秒のカウントダウン後に判定します。
        </p>
      </div>

      <div className="scoreboard">
        <div className="score">
          <div className="num">{scoreP}</div>
          <div className="lbl">あなた</div>
        </div>
        <div className="score">
          <div className="num">{scoreA}</div>
          <div className="lbl">AI</div>
        </div>
        <div className="score">
          <div className="num" style={{ fontSize: 20 }}>
            {lastResult}
          </div>
          <div className="lbl">直近の結果</div>
        </div>
      </div>

      <div className="layout">
        <div className="controls">
          <h3>🎛️ コントロール</h3>
          <button className="btn primary" onClick={startRound} disabled={running}>
            ✊ じゃんけん開始！
          </button>
          <button className="btn" onClick={resetScore}>
            🔁 スコアリセット
          </button>
          <div className="divider" />
          {running ? (
            <span className="badge green">🟢 判定中</span>
          ) : (
            <span className="badge grey">⚪ 待機中</span>
          )}
        </div>
        <div>
          <Stage
            videoRef={videoRef}
            canvasRef={canvasRef}
            status={status}
            error={error}
            placeholder="「✊ じゃんけん開始！」を押してね"
          />
          <div className="hint-box">
            <b>まず実装しよう：</b> <code>src/lib/jankenLogic.js</code> の{" "}
            <code>classifyHandGesture()</code> と <code>decideWinner()</code> の穴埋めを
            完成させないと、AIが手を判定できません。
          </div>
        </div>
      </div>
    </div>
  );
}
