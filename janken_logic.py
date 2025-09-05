# janken_logic.py
# このファイルに、じゃんけんの判定ロジックを実装しよう！

def classify_hand_gesture(hand_landmarks, image_shape):

    # --- ここから実装してみよう ---

    # ヒント1: 画像の高さを取得
    h, w = image_shape[:2]

    # ヒント2: 指の先端と第二関節のランドマーク番号
    # 人差し指, 中指, 薬指, 小指 の先端と第二関節
    TIP_IDS = [8, 12, 16, 20]
    PIP_IDS = [6, 10, 14, 18]

    # ヒント3: 「伸びている」指の本数を数えるための変数を用意
    count = 0

    # ヒント4: forループを使って4本の指が伸びているかチェック
    # 指の先端のy座標が、第二関節のy座標よりも小さい（画面の上側にある）場合、「伸びている」と判断できる
    for tip_id, pip_id in zip(TIP_IDS, PIP_IDS):
        tip_y = hand_landmarks.landmark[tip_id].y
        pip_y = hand_landmarks.landmark[pip_id].y

        if tip_y < pip_y:
            count += 1

    # ヒント5: 伸びている指の本数に応じて、返す手を変える
    # 4本以上伸びていたら「パー」
    if count >= 4:
        return "PA"
    # 2本伸びていたら「チョキ」
    elif count == 2:
        return "CHOKI"
    # 1本以下なら「グー」
    elif count <= 1:
        return "GU"

    return "UNKNOWN" # どれにも当てはまらない場合


def decide_winner(player_hand, ai_hand):
    """
    プレイヤーの手とAIの手を比べて、勝敗を判定する関数。

    引数:
        player_hand: プレイヤーの手 ("GU", "CHOKI", "PA")
        ai_hand: AIの手 ("GU", "CHOKI", "PA")

    戻り値:
        プレイヤーから見た結果 "WIN", "LOSE", "DRAW" のいずれかの文字列
    """

    # --- ここから実装してみよう ---

    # ヒント1: あいこ（DRAW）になる条件は？
    if player_hand == ai_hand:
        return "DRAW"

    # ヒント2: プレイヤーが勝つ（WIN）組み合わせを考える
    # Pythonでは (A and B) or (C and D) のように条件を組み合わせられる
    if (player_hand == "GU" and ai_hand == "CHOKI") or \
       (player_hand == "CHOKI" and ai_hand == "PA") or \
       (player_hand == "PA" and ai_hand == "GU"):
        return "WIN"

    # ヒント3: 上のいずれにも当てはまらなければ、プレイヤーの負け（LOSE）
    else:
        return "LOSE"