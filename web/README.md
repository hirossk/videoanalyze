# AIを使った動画分析 体験入学コース（React版）

Python（Streamlit）版を React + ブラウザ版 MediaPipe に移植したものです。
**1つのアプリの中で Step1〜6・じゃんけんを左メニューから切り替え**られます。
毎回 `streamlit run xxx.py` を打ち替える必要はありません。

## 動かし方

```bash
cd web
npm install      # 最初の1回だけ
npm run dev      # 開発サーバー起動（ブラウザが自動で開きます）
```

表示された URL（通常 http://localhost:5173 ）を開き、左メニューでステップを切り替えます。
カメラを使うステップでは、ブラウザの「カメラ使用許可」を許可してください。

> 初回はAIモデル（数MB）をネットからダウンロードするため、最初の起動だけ少し時間がかかります。

## ステップと「穴埋め」ファイル

| メニュー | ファイル | 穴埋めの場所 |
|---|---|---|
| Step 1 はじめてのWebアプリ | `src/steps/Step1.jsx` | タイトル/入力欄/あいさつ選択肢 |
| Step 2 カメラ表示 | `src/steps/Step2.jsx` | `toGray()` / `flipVertical()` |
| Step 3 顔の検出 | `src/steps/Step3.jsx` | 枠を描く / `faceEmoji` |
| Step 4 手の検出 | `src/steps/Step4.jsx` | 骨格描画 / `fingerNames` / ラベル表示 |
| Step 5 サングラス | `src/steps/Step5.jsx` | メッシュ / 目印 / `drawImage` 合成 |
| Step 6 ストレッチ | `src/steps/Step6.jsx` | `EAR` → `SHOULDER` |
| 応用 じゃんけんAI | `src/lib/jankenLogic.js` | `classifyHandGesture()` / `decideWinner()` |
| 体験デモ オープンキャンパス | `src/steps/OpenCampusDemo.jsx` | （完成版・穴埋めなし） |

各ファイルの中に `★穴埋め` というコメントがあります。Python版と同じように
コメントを外したり `""` を書き換えたりして完成させましょう。

> **体験デモ（オープンキャンパス）** は `opencampus_demo.py` を移植した完成版デモです。
> 顔にサングラス😎/猫耳🐱、手のサイン（✌✋✊👍🦊）でエフェクト、両手バンザイ🙌/ハート🫶、
> 全身Tポーズ🕺、効果音までブラウザだけで動きます（穴埋めなし）。

## メインの切り替えについて

`src/App.jsx` の `MENU` 配列が「切り替えメニュー」の正体です。
ここに1行足すだけで、新しいステップをメニューに追加できます。

## 仕組み（共通部品）

- `src/hooks/useCameraLoop.js` … カメラ起動〜毎フレーム処理〜停止の共通ループ
- `src/lib/vision.js` … MediaPipe（顔・手・姿勢・FaceMesh）の準備
- `src/components/` … 映像の額縁とコントロールボタン
