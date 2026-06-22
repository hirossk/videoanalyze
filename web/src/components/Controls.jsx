// サイドの「コントロール」エリア。起動／停止ボタンと状態ラベルをまとめた部品。
export function Controls({ running, onStart, onStop, startLabel = "▶️ AIカメラ起動" }) {
  return (
    <div className="controls">
      <h3>🎛️ コントロール</h3>
      <button className="btn primary" onClick={onStart} disabled={running}>
        {startLabel}
      </button>
      <button className="btn" onClick={onStop} disabled={!running}>
        🛑 停止
      </button>
      <div className="divider" />
      {running ? (
        <span className="badge green">🟢 実行中</span>
      ) : (
        <span className="badge grey">⚪ 停止中</span>
      )}
    </div>
  );
}
