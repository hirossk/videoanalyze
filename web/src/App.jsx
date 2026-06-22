import { useState } from "react";
import Step1 from "./steps/Step1.jsx";
import Step2 from "./steps/Step2.jsx";
import Step3 from "./steps/Step3.jsx";
import Step4 from "./steps/Step4.jsx";
import Step5 from "./steps/Step5.jsx";
import Step6 from "./steps/Step6.jsx";
import Janken from "./steps/Janken.jsx";

// ★ここが「メインの切り替え」の中心 ★
// 左メニューの一覧。ここの並びを変えるだけで切り替えメニューが変わります。
const MENU = [
  { id: "step1", label: "Step 1 ── はじめてのWebアプリ", icon: "🎈", Component: Step1 },
  { id: "step2", label: "Step 2 ── カメラを表示しよう", icon: "📷", Component: Step2 },
  { id: "step3", label: "Step 3 ── 顔を検出しよう", icon: "🙂", Component: Step3 },
  { id: "step4", label: "Step 4 ── 手を検出しよう", icon: "🖐️", Component: Step4 },
  { id: "step5", label: "Step 5 ── サングラスをかけよう", icon: "😎", Component: Step5 },
  { id: "step6", label: "Step 6 ── ストレッチカウンター", icon: "🤸", Component: Step6 },
  { id: "janken", label: "応用 ── じゃんけんAI", icon: "✊", Component: Janken },
];

export default function App() {
  // いま選ばれているステップの id を覚えておく「いれもの」
  const [current, setCurrent] = useState("step1");

  // 選ばれた id に対応するコンポーネントを取り出す
  const active = MENU.find((m) => m.id === current);
  const ActiveComponent = active.Component;

  return (
    <div className="app">
      <aside className="sidebar">
        <h1 className="brand">🎈 AI動画分析コース</h1>
        <p className="brand-sub">メニューから選んで切り替え</p>
        <nav>
          {MENU.map((m) => (
            <button
              key={m.id}
              className={"nav-item" + (m.id === current ? " active" : "")}
              onClick={() => setCurrent(m.id)}
            >
              <span className="nav-icon">{m.icon}</span>
              <span>{m.label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main className="content">
        {/* key を付けることで、切り替え時に前のステップを完全にリセット（カメラも停止）する */}
        <ActiveComponent key={current} />
      </main>
    </div>
  );
}
