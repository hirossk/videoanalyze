import { useState } from "react";

// Step 1 ── はじめてのWebアプリ
// テーマ：名前を入力するとあいさつしてくれるページを作る。
export default function Step1() {
  // 入力された名前をしまっておく「いれもの（変数）」
  const [name, setName] = useState("");
  // 選ばれたあいさつをしまっておく「いれもの」
  const [greeting, setGreeting] = useState("こんにちは！");
  // ボタンが押されたかどうかの合図
  const [submitted, setSubmitted] = useState(false);

  // ★穴埋め④：あいさつの選択肢。"" の部分を「おはようございます！」などに書き換えよう
  const greetings = ["こんにちは！", "", ""];

  // フォームが送信されたとき（ボタンが押されたとき）の処理
  function handleSubmit(e) {
    e.preventDefault();
    setSubmitted(true);
  }

  return (
    <div>
      <div className="step-header">
        {/* ★穴埋め①：下の <h2> と <p> のコメントを外して、タイトルと説明文を表示しよう */}
        {/* <h2>🎈 はじめてのWebアプリ</h2> */}
        {/* <p className="caption">名前を入れてボタンを押してみてね！</p> */}
      </div>

      <form className="form-card" onSubmit={handleSubmit}>
        <div className="field">
          {/* ★穴埋め②：名前入力欄。下のコメントを外して入力箱を作ろう */}
          
          {/* <label>ここに名前を入力してください</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="なまえ"
          /> */}
         
        </div>

        {/* <div className="field">
          <label>あいさつを選んでください</label>
          <select value={greeting} onChange={(e) => setGreeting(e.target.value)}>
            {greetings
              .filter((g) => g !== "") // 空の選択肢は表示しない
              .map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
          </select>
        </div> */}

        {/* ★穴埋め③：送信ボタン。これでフォームが送られて submitted が true になる */}
        {/* <button className="btn primary" type="submit">
          あいさつする
        </button> */}
      </form>

      {/* ボタンが押された後だけ、ここから下が表示される */}
      {submitted &&
        (name ? (
          // 名前が入っていたら、緑のメッセージを表示
          <div className="message success">
            {name}さん、{greeting}
          </div>
        ) : (
          // 名前が空っぽだったら、黄色の注意メッセージを表示
          <div className="message warning">名前が入力されていません。</div>
        ))}

      <div className="hint-box">
        <b>やること：</b> ①タイトル/説明の <code>{"{/* */}"}</code> を外す → ②名前入力欄の
        コメントを外す → ③ボタンはもう動く → ④<code>greetings</code> の{" "}
        <code>""</code> をあいさつの言葉に書き換える。
      </div>
    </div>
  );
}
