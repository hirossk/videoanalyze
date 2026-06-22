// カメラ映像を表示する「額縁」。各ステップで共通して使う見た目の部品。
export function Stage({ videoRef, canvasRef, status, error, placeholder }) {
  return (
    <div className="stage">
      {/* 映像のもとになる <video>。画面には出さず、裏で動かしてキャンバスに描く */}
      <video ref={videoRef} playsInline muted className="hidden-video" />

      {/* 実際に映像とAIの結果を描く場所 */}
      <canvas ref={canvasRef} className="view" />

      {/* カメラが止まっているときの案内 */}
      {status === "idle" && (
        <div className="overlay">
          <p>{placeholder || "「▶️ カメラ起動」ボタンを押してください。"}</p>
        </div>
      )}
      {status === "loading" && (
        <div className="overlay">
          <p>🧠 AIモデルを読み込み中…（初回は少し時間がかかります）</p>
        </div>
      )}
      {status === "error" && (
        <div className="overlay error">
          <p>⚠️ エラー: {error}</p>
          <p className="small">カメラの使用を許可しているか確認してください。</p>
        </div>
      )}
    </div>
  );
}
