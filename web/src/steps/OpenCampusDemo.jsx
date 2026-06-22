import { useEffect, useRef, useState } from "react";
import { useCameraLoop } from "../hooks/useCameraLoop.js";
import { Stage } from "../components/Stage.jsx";
import {
  createFaceLandmarker,
  createHandLandmarker,
  createPoseLandmarker,
  DrawingUtils,
  HandLandmarker,
} from "../lib/vision.js";

// ============================================================
//  🎓 AI画像認識 体験デモ（オープンキャンパス用）の React 版
//  opencampus_demo.py（Streamlit）を、ブラウザ版 MediaPipe に移植したもの。
//   ・顔 …… サングラス😎 / 猫耳🐱 を自動装着
//   ・手 …… ✌ピース / ✋パー / ✊グー / 👍サムズアップ / 🦊きつね を判定
//   ・両手/全身 …… 🙌バンザイ / 🫶ハート / 🕺Tポーズ
// ============================================================

// --- 設定（Python版の定数と同じ意味）---
const SUNG_WIDTH_FACTOR = 2.2; // サングラスの幅を「両目の間」の何倍にするか
const FINGER_EXT_MARGIN = 0.12; // 指が「伸びている」とみなす余裕
const THUMB_OPEN_RATIO = 0.5; // 親指が「開いている」とみなす距離の比率
const GESTURE_SMOOTH_N = 5; // 直近Nフレームの多数決でブレを抑える
const GESTURE_MIN_VOTES = 3; // N回中、最低この回数そろって初めて確定

// 日本語ラベルとテーマカラー（CSS の rgb）と絵文字
const GESTURE_INFO = {
  PEACE: { label: "ピース！", color: "rgb(60,120,255)", emoji: "✌" },
  PA: { label: "パー！", color: "rgb(255,200,60)", emoji: "✋" },
  GU: { label: "グー！", color: "rgb(255,80,80)", emoji: "✊" },
  GOOD: { label: "いいね！ GOOD", color: "rgb(90,220,60)", emoji: "👍" },
  FOX: { label: "きつね！", color: "rgb(255,150,30)", emoji: "🦊" },
};

// ------------------------------------------------------------
// 効果音（WebAudio）。グー/パー/チョキにそれぞれ違う和音を割り当てる。
// ------------------------------------------------------------
let _audioCtx = null;
function ensureAudio() {
  if (!_audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (Ctx) _audioCtx = new Ctx();
  }
  return _audioCtx;
}
// ジェスチャーごとの「メロディ（音の並び）」。
//   at:鳴り始め(秒) / f:周波数 / to:終わりの周波数(しゃくり上げ等) /
//   dur:長さ / type:音色 / vol:音量
// 短いフレーズにすることで「何のサインか」を耳でハッキリ区別できる。
const GESTURE_SOUNDS = {
  // グー：低くて短い「ドゥンッ」（パンチのある下降音）
  GU: [{ at: 0, f: 200, to: 90, dur: 0.22, type: "sawtooth", vol: 0.4 }],
  // パー：明るく上がっていく「ドミソド」（成功チャイム）
  PA: [
    { at: 0.0, f: 523 },
    { at: 0.09, f: 659 },
    { at: 0.18, f: 784 },
    { at: 0.27, f: 1047, dur: 0.26 },
  ],
  // チョキ（ピース）：キラッと上がる2音「ピロリン」
  PEACE: [
    { at: 0.0, f: 784, type: "sine" },
    { at: 0.1, f: 1175, dur: 0.24, type: "sine" },
  ],
  // サムズアップ：レベルアップ風の3音上昇
  GOOD: [
    { at: 0.0, f: 659 },
    { at: 0.09, f: 988 },
    { at: 0.18, f: 1319, dur: 0.26 },
  ],
  // きつね：コミカルに跳ねる3音
  FOX: [
    { at: 0.0, f: 587, type: "square", vol: 0.25 },
    { at: 0.1, f: 880, type: "square", vol: 0.25 },
    { at: 0.2, f: 740, dur: 0.22, type: "square", vol: 0.25 },
  ],
};

// 音の並び（メロディ）を鳴らす。1音ずつエンベロープ付きで再生する。
function playSeq(notes) {
  const ac = ensureAudio();
  if (!ac) return;
  if (ac.state === "suspended") ac.resume();
  const t0 = ac.currentTime;
  for (const n of notes) {
    const start = t0 + (n.at ?? 0);
    const dur = n.dur ?? 0.16;
    const vol = n.vol ?? 0.3;
    const osc = ac.createOscillator();
    const gain = ac.createGain();
    osc.type = n.type ?? "triangle";
    osc.frequency.setValueAtTime(n.f, start);
    if (n.to) osc.frequency.exponentialRampToValueAtTime(n.to, start + dur);
    // アタックを速く、減衰をなめらかに（プツッと鳴らない）
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(vol, start + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);
    osc.connect(gain);
    gain.connect(ac.destination);
    osc.start(start);
    osc.stop(start + dur + 0.03);
  }
}

// ------------------------------------------------------------
// ジェスチャー判定（classify_gesture の移植）
// ------------------------------------------------------------
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function fingersExtended(lm, handSize) {
  const wrist = lm[0];
  const tips = [8, 12, 16, 20];
  const pips = [6, 10, 14, 18];
  return tips.map((tip, i) => {
    const diff = (dist(lm[tip], wrist) - dist(lm[pips[i]], wrist)) / handSize;
    return diff > FINGER_EXT_MARGIN;
  });
}
function classifyGesture(lm) {
  const wrist = lm[0];
  const handSize = dist(wrist, lm[9]) + 1e-6;
  const [idx, mid, ring, pinky] = fingersExtended(lm, handSize);
  const nExt = [idx, mid, ring, pinky].filter(Boolean).length;
  const thumbOpen = dist(lm[4], lm[2]) / handSize > THUMB_OPEN_RATIO;
  const thumbUp = lm[4].y < lm[2].y - 0.04 && lm[4].y < wrist.y;

  if (idx && mid && ring && pinky && thumbOpen) return "PA";
  if (idx && pinky && !mid && !ring) return "FOX";
  if (idx && mid && !ring && !pinky) return "PEACE";
  if (nExt === 0 && thumbOpen && thumbUp) return "GOOD";
  if (nExt === 0) return "GU";
  return "UNKNOWN";
}

// 配列の多数決（票数つき）
function topVote(arr) {
  if (!arr.length) return [null, 0];
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
  return [best, bestN];
}

// ------------------------------------------------------------
// 描画ヘルパー
// ------------------------------------------------------------
function drawLabel(ctx, text, x, y, size, color) {
  ctx.font = `bold ${size}px "Noto Sans JP", sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.lineWidth = Math.max(2, size * 0.08);
  ctx.strokeStyle = "#000";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function drawStar(ctx, x, y, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  for (let a = 0; a < 3; a++) {
    const ang = (a * Math.PI) / 3;
    const c = Math.cos(ang) * 9;
    const s = Math.sin(ang) * 9;
    ctx.beginPath();
    ctx.moveTo(x - c, y - s);
    ctx.lineTo(x + c, y + s);
    ctx.stroke();
  }
}

// ジェスチャーに合わせて手の近くに派手なエフェクトを描く
function drawGestureEffect(ctx, gesture, cx, cy, t) {
  const info = GESTURE_INFO[gesture];
  // 広がるリング
  const phase = (t * 1.5) % 1.0;
  ctx.lineWidth = 3;
  ctx.strokeStyle = info.color;
  for (let k = 0; k < 3; k++) {
    const r = 40 + (((phase + k / 3.0) % 1.0) * 90);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  // 回る星
  for (let k = 0; k < 6; k++) {
    const ang = t * 2 + k * ((2 * Math.PI) / 6);
    drawStar(ctx, cx + Math.cos(ang) * 75, cy + Math.sin(ang) * 75, info.color);
  }
  // 日本語ラベルと絵文字
  drawLabel(ctx, info.label, cx - 90, cy - 150, 56, info.color);
  ctx.font = "70px serif";
  ctx.textAlign = "left";
  ctx.fillText(info.emoji, cx + 70, cy - 100);
}

// バンザイの紙吹雪
const CONFETTI_PALETTE = [
  "rgb(255,200,60)", "rgb(255,80,80)", "rgb(90,220,60)",
  "rgb(60,120,255)", "rgb(200,80,255)", "rgb(255,255,60)",
];
function drawConfetti(ctx, t, W, H) {
  for (let i = 0; i < 48; i++) {
    const x = (i * 137) % W;
    const speed = 160 + ((i * 53) % 180);
    const y = ((t * speed + i * 71) % (H + 40)) - 20;
    ctx.fillStyle = CONFETTI_PALETTE[i % CONFETTI_PALETTE.length];
    ctx.fillRect(x, y, 8, 14);
  }
}

// ハートマーク（2つの円＋三角）
function drawHeart(ctx, cx, cy, size, color = "rgb(255,80,80)") {
  const r = Math.max(6, size / 4);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx - r, cy - r / 2, r, 0, Math.PI * 2);
  ctx.arc(cx + r, cy - r / 2, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(cx - 2 * r, cy - r / 4);
  ctx.lineTo(cx + 2 * r, cy - r / 4);
  ctx.lineTo(cx, cy + 2 * r);
  ctx.closePath();
  ctx.fill();
}

// 顔の傾き（両目を結ぶ線の角度・ラジアン）
function eyeAngle(lm, W, H) {
  const le = lm[33];
  const re = lm[263];
  return Math.atan2((re.y - le.y) * H, (re.x - le.x) * W);
}

// サングラスを顔の傾きに合わせて合成
function drawSunglasses(ctx, lm, img, W, H) {
  if (!img) return;
  const le = lm[33];
  const re = lm[263];
  const lx = le.x * W;
  const ly = le.y * H;
  const rx = re.x * W;
  const ry = re.y * H;
  const eyeW = Math.hypot(rx - lx, ry - ly);
  const width = eyeW * SUNG_WIDTH_FACTOR;
  if (width < 10) return;
  const height = width * (img.height / img.width);
  const cx = (lx + rx) / 2;
  const cy = (ly + ry) / 2;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(eyeAngle(lm, W, H));
  ctx.drawImage(img, -width / 2, -height / 2, width, height);
  ctx.restore();
}

// 1つの猫耳の輪郭をたどる（丸い先端＆ふくらみのある曲線）。
// ローカル座標：(cx, baseY) が耳の根元の中心、上方向が -y。
function traceEar(ctx, cx, baseY, w, h, lean) {
  const tipX = cx + lean;
  ctx.beginPath();
  ctx.moveTo(cx - w / 2, baseY);
  // 左の辺 → 先端へ（外側にふくらむ）
  ctx.quadraticCurveTo(cx - w * 0.5, baseY - h * 0.55, tipX - w * 0.12, baseY - h);
  // 丸い先っぽ
  ctx.quadraticCurveTo(tipX, baseY - h * 1.12, tipX + w * 0.12, baseY - h);
  // 右の辺 → 根元へ
  ctx.quadraticCurveTo(cx + w * 0.5, baseY - h * 0.55, cx + w / 2, baseY);
  ctx.closePath();
}

// カッコいい猫耳（左右）を頭の上に合成する
function drawCatEars(ctx, lm, W, H) {
  const lf = lm[234];
  const rf = lm[454];
  const faceW = Math.hypot((rf.x - lf.x) * W, (rf.y - lf.y) * H);
  const w = faceW * 1.5;
  if (w < 20) return;
  const fx = lm[10].x * W;
  const fy = lm[10].y * H;

  const dx = w * 0.3; // 左右の耳の間隔
  const ew = w * 0.42; // 耳の幅
  const eh = w * 0.66; // 耳の高さ

  ctx.save();
  ctx.translate(fx, fy);
  ctx.rotate(eyeAngle(lm, W, H));
  ctx.translate(0, -w * 0.08); // 少し頭の上へ
  ctx.lineJoin = "round";

  for (const side of [-1, 1]) {
    const cx = side * dx;
    const lean = side * ew * 0.16; // 先っぽを外側へ倒す＝猫らしい

    // 外側（毛）：上が暗く下が明るいグラデーション＋縁取り
    const fur = ctx.createLinearGradient(0, -eh, 0, 0);
    fur.addColorStop(0, "#211c28");
    fur.addColorStop(1, "#4d4554");
    traceEar(ctx, cx, 0, ew, eh, lean);
    ctx.fillStyle = fur;
    ctx.fill();
    ctx.lineWidth = Math.max(2, w * 0.04);
    ctx.strokeStyle = "#13101a";
    ctx.stroke();

    // 内耳（ピンク）：少し上にずらして小さく
    const pink = ctx.createLinearGradient(0, -eh * 0.66, 0, -eh * 0.05);
    pink.addColorStop(0, "#ff6fa5");
    pink.addColorStop(1, "#ffc2d8");
    traceEar(ctx, cx + lean * 0.4, -eh * 0.08, ew * 0.5, eh * 0.66, lean * 0.6);
    ctx.fillStyle = pink;
    ctx.fill();
  }
  ctx.restore();
}

// 両手の位置から「バンザイ」「ハート」を判定
function detectTwoHandPose(centers, faceTopY, W) {
  if (centers.length < 2) return null;
  const [[x1, y1], [x2, y2]] = centers;
  if (y1 < faceTopY && y2 < faceTopY) return "BANZAI";
  const close = Math.hypot(x2 - x1, y2 - y1) < W * 0.28;
  const below = y1 > faceTopY && y2 > faceTopY;
  if (close && below) return "HEART";
  return null;
}

// 全身の関節から「Tポーズ」「バンザイ」を判定
function classifyBodyPose(lm) {
  const shY = (lm[11].y + lm[12].y) / 2;
  const spread = Math.abs(lm[15].x - lm[16].x);
  const level = Math.abs(lm[15].y - shY) < 0.15 && Math.abs(lm[16].y - shY) < 0.15;
  if (spread > 0.55 && level) return "TPOSE";
  if (lm[15].y < lm[0].y && lm[16].y < lm[0].y) return "BANZAI";
  return null;
}

// 案内バナー（上下）
function drawBanner(ctx, fps, W, H) {
  ctx.save();
  ctx.fillStyle = "rgba(30,20,20,0.5)";
  ctx.fillRect(0, 0, W, 52);
  ctx.fillRect(0, H - 44, W, 44);
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#fff";
  ctx.font = 'bold 26px "Noto Sans JP", sans-serif';
  ctx.fillText("AI画像認識 体験デモ", 16, 26);
  ctx.fillStyle = "rgb(120,230,255)";
  ctx.font = '20px "Noto Sans JP", sans-serif';
  ctx.fillText("ピース・パー・グー・サムズアップ・きつねを見せてね！", 16, H - 22);
  ctx.fillStyle = "#fff";
  ctx.textAlign = "right";
  ctx.font = "20px sans-serif";
  ctx.fillText(`FPS:${fps.toFixed(1)}`, W - 14, 26);
  ctx.restore();
}

// ============================================================
//  React コンポーネント
// ============================================================
export default function OpenCampusDemo() {
  const [running, setRunning] = useState(false);
  const [deco, setDeco] = useState("both"); // sunglasses / catears / both / none
  const [showFun, setShowFun] = useState(true); // バンザイ/ハート
  const [usePose, setUsePose] = useState(false); // 全身ポーズ（重め）
  const [enableSound, setEnableSound] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState(false);

  const showSunglasses = deco === "sunglasses" || deco === "both";
  const showCatears = deco === "catears" || deco === "both";

  // サングラス画像を読み込んでおく
  const glassesRef = useRef(null);
  useEffect(() => {
    const img = new Image();
    img.src = "/glasses/glasses1.png";
    img.onload = () => (glassesRef.current = img);
  }, []);

  // 毎フレームで使い回す状態（再描画を避けるため ref に置く）
  const buffersRef = useRef([[], []]); // 手ごとのジェスチャー履歴
  const lastSoundRef = useRef([null, null]); // 手ごとに最後に鳴らしたサイン
  const fpsRef = useRef(0);
  const prevTimeRef = useRef(0);

  const { videoRef, canvasRef, status, error } = useCameraLoop({
    running,
    // 顔(2)・手(2)・姿勢 の専門家をまとめて準備する
    setup: async () => {
      buffersRef.current = [[], []];
      lastSoundRef.current = [null, null];
      prevTimeRef.current = performance.now() / 1000;
      const [face, hands, pose] = await Promise.all([
        createFaceLandmarker(2),
        createHandLandmarker(2),
        createPoseLandmarker(),
      ]);
      return { face, hands, pose };
    },

    process: (ctx, video, det) => {
      if (!det) return;
      const W = ctx.canvas.width;
      const H = ctx.canvas.height;
      const ts = performance.now();
      const t = ts / 1000;
      const drawing = new DrawingUtils(ctx);

      const faceRes = det.face.detectForVideo(ctx.canvas, ts);
      const handRes = det.hands.detectForVideo(ctx.canvas, ts);
      const poseRes = usePose ? det.pose.detectForVideo(ctx.canvas, ts) : null;

      const faces = faceRes.faceLandmarks ?? [];

      // 顔の上端（バンザイ判定などに使う）
      let faceTopY = H * 0.25;
      if (faces.length) {
        faceTopY = Math.min(...faces.map((f) => f[10].y * H));
      }

      // --- 顔の飾り（サングラス／猫耳）---
      if (showSunglasses || showCatears) {
        for (const lm of faces) {
          if (showCatears) drawCatEars(ctx, lm, W, H);
          if (showSunglasses) drawSunglasses(ctx, lm, glassesRef.current, W, H);
        }
      }

      // --- 手：ジェスチャー判定 → エフェクト ---
      const handCenters = [];
      const handLandmarks = handRes.landmarks ?? [];
      if (handLandmarks.length) {
        for (let i = 0; i < handLandmarks.length && i < 2; i++) {
          const lm = handLandmarks[i];
          if (showLandmarks) {
            drawing.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, {
              color: "#00ff00",
              lineWidth: 1,
            });
            drawing.drawLandmarks(lm, { color: "#0080ff", radius: 2 });
          }
          const cx = lm[9].x * W;
          const cy = lm[9].y * H;
          handCenters.push([cx, cy]);

          const g = classifyGesture(lm);
          const buf = buffersRef.current[i];
          if (g !== "UNKNOWN") {
            buf.push(g);
            if (buf.length > GESTURE_SMOOTH_N) buf.shift();
          }
          const [cand, votes] = topVote(buf);
          if (cand && votes >= GESTURE_MIN_VOTES) {
            drawGestureEffect(ctx, cand, cx, cy, t);
            if (cand !== lastSoundRef.current[i]) {
              if (enableSound && GESTURE_SOUNDS[cand]) playSeq(GESTURE_SOUNDS[cand]);
              lastSoundRef.current[i] = cand;
            }
          }
        }
      } else {
        buffersRef.current = [[], []];
        lastSoundRef.current = [null, null];
      }

      // --- おもしろポーズ（両手）---
      if (showFun) {
        const pose2 = detectTwoHandPose(handCenters, faceTopY, W);
        if (pose2 === "BANZAI") {
          drawConfetti(ctx, t, W, H);
          drawLabel(ctx, "バンザイ！", W / 2 - 130, 110, 80, "rgb(255,255,60)");
          ctx.font = "80px serif";
          ctx.fillText("🙌", W / 2 + 120, 110);
        } else if (pose2 === "HEART") {
          const mx = (handCenters[0][0] + handCenters[1][0]) / 2;
          const my = (handCenters[0][1] + handCenters[1][1]) / 2;
          drawHeart(ctx, mx, my - 30, 70, "rgb(255,120,200)");
          drawLabel(ctx, "だいすき♡", mx - 110, my - 170, 56, "rgb(255,120,200)");
        }
      }

      // --- 全身ポーズ（Tポーズ等）---
      if (poseRes && poseRes.landmarks?.[0]) {
        const bp = classifyBodyPose(poseRes.landmarks[0]);
        if (bp === "TPOSE") {
          drawLabel(ctx, "Tポーズ！", W / 2 - 130, 160, 80, "rgb(90,220,60)");
        } else if (bp === "BANZAI" && !showFun) {
          drawConfetti(ctx, t, W, H);
          drawLabel(ctx, "バンザイ！", W / 2 - 130, 160, 80, "rgb(255,255,60)");
        }
      }

      // --- FPS 計算 & 案内バナー ---
      const dt = t - prevTimeRef.current;
      prevTimeRef.current = t;
      if (dt > 0) fpsRef.current = 0.9 * fpsRef.current + 0.1 * (1.0 / dt);
      drawBanner(ctx, fpsRef.current, W, H);
    },
  });

  return (
    <div>
      <div className="step-header">
        <h2>🎓 AI画像認識 体験デモ</h2>
        <p className="caption">
          😎 顔を見つけると飾り／✌ ✋ ✊ 👍 🦊 の手のサインや 🙌 🫶 のポーズでエフェクトが出ます！
        </p>
      </div>

      <div className="layout">
        <div className="controls">
          <h3>🎛️ コントロール</h3>
          <button className="btn primary" onClick={() => setRunning(true)} disabled={running}>
            ▶️ スタート
          </button>
          <button className="btn" onClick={() => setRunning(false)} disabled={!running}>
            ⏹️ 停止
          </button>
          <div className="divider" />

          <div className="field">
            <label>😎 顔の飾り</label>
            <select value={deco} onChange={(e) => setDeco(e.target.value)}>
              <option value="sunglasses">サングラス😎</option>
              <option value="catears">猫耳🐱</option>
              <option value="both">サングラス+猫耳</option>
              <option value="none">なし</option>
            </select>
          </div>

          <label className="check">
            <input type="checkbox" checked={showFun} onChange={(e) => setShowFun(e.target.checked)} />
            🎉 おもしろポーズ（バンザイ/ハート）
          </label>
          <label className="check">
            <input type="checkbox" checked={usePose} onChange={(e) => setUsePose(e.target.checked)} />
            🕺 全身ポーズも判定（Tポーズ等・少し重め）
          </label>
          <label className="check">
            <input type="checkbox" checked={enableSound} onChange={(e) => setEnableSound(e.target.checked)} />
            🔊 効果音（グー/パー/チョキ）
          </label>
          <label className="check">
            <input type="checkbox" checked={showLandmarks} onChange={(e) => setShowLandmarks(e.target.checked)} />
            🖐️ 手の骨格を表示
          </label>

          <div className="divider" />
          {running ? (
            <span className="badge green">🟢 実行中</span>
          ) : (
            <span className="badge grey">⚪ 停止中</span>
          )}
        </div>

        <div>
          <Stage
            videoRef={videoRef}
            canvasRef={canvasRef}
            status={status}
            error={error}
            placeholder="「▶️ スタート」を押すとカメラが起動します。"
          />
          <div className="hint-box">
            <b>遊び方：</b> ①カメラに顔を向けると飾りがつくよ😎🐱 → ②手のサインを見せてね（✌✋✊👍🦊）
            → ③🙌 両手を上げてバンザイ / 🫶 両手を近づけてハート / 🕺 Tポーズ（全身ポーズON時）。
          </div>
        </div>
      </div>
    </div>
  );
}
