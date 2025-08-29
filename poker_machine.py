"""
我和调用的API玩德扑
note: 记录n个玩家的下注
提供下注端口
"""
import random
import itertools
from typing import List, Tuple, Optional, Dict

# =========================
# 配置（可自行修改）
# =========================
SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 5000
ALLOW_INVALID_INPUT_RETRY = True  # 输入错了反复提示

# =========================
# 牌与玩家
# =========================
RANKS = list(range(2, 15))  # 2..14(=A)
SUITS = ["♠", "♥", "♦", "♣"]
RANK_TO_STR = {11: "J", 12: "Q", 13: "K", 14: "A"}
STR_TO_RANK = {**{str(i): i for i in range(2, 11)},
               **{"J": 11, "Q": 12, "K": 13, "A": 14}}

def rank_to_str(r: int) -> str:
    return RANK_TO_STR.get(r, str(r))

class Card:
    def __init__(self, rank: int, suit: str):
        self.rank = rank
        self.suit = suit
    def __repr__(self) -> str:
        return f"{rank_to_str(self.rank)}{self.suit}"

class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for r in RANKS for s in SUITS]
    def shuffle(self):
        random.shuffle(self.cards)
    def deal(self, n: int) -> List[Card]:
        d = self.cards[:n]
        self.cards = self.cards[n:]
        return d
    def burn(self):
        if self.cards:
            self.cards.pop(0)

class Player:
    def __init__(self, name: str, stack: int):
        self.name = name
        self.stack = stack
        self.hole: List[Card] = []
        self.is_button = False
        # 本轮需要跟注的金额（针对当前轮的已投入差额）
        self.to_call = 0
        # 标记是否全下
        self.all_in = False
        self.has_folded = False

    def reset_for_new_hand(self):
        self.hole = []
        self.to_call = 0
        self.all_in = False
        self.has_folded = False

# =========================
# 评牌（5张）与 7 取 5
# =========================
# Hand category：8=StraightFlush, 7=FourKind, 6=FullHouse, 5=Flush, 4=Straight, 3=Trips, 2=TwoPair, 1=OnePair, 0=HighCard
def evaluate_5(cards: List[Card]) -> Tuple[int, List[int]]:
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    # 计数
    counts: Dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # 是否同花
    is_flush = False
    flush_suit = None
    suit_counts: Dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
        if suit_counts[s] == 5:
            is_flush = True
            flush_suit = s
    # 是否顺子（处理 A 作为 1 的情况）
    def straight_high(ranks_desc: List[int]) -> Optional[int]:
        # 去重保序
        r = []
        for x in ranks_desc:
            if x not in r:
                r.append(x)
        # A-5 顺子支持：把 A 当作 1 加入末尾
        if 14 in r:
            r.append(1)
        # 连续5张取最大高点
        for i in range(len(r) - 4):
            window = r[i:i+5]
            # 等差为 -1
            if all(window[j] - 1 == window[j+1] for j in range(4)):
                return window[0]  # 最高张
        return None

    # Straight / Straight flush
    sf_high = None
    if is_flush:
        flush_cards = sorted([c for c in cards if c.suit == flush_suit], key=lambda x: x.rank, reverse=True)
        sf_high = straight_high([c.rank for c in flush_cards])

    if sf_high is not None:
        return (8, [sf_high])  # Straight Flush

    # 四条 / 葫芦 / 三条 / 两对 / 一对 分桶
    by_count = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # 按数量（多到少）、再按点数（大到小）
    if by_count[0][1] == 4:
        # 四条: [四条点, 踢脚]
        four = by_count[0][0]
        kicker = max([r for r in ranks if r != four])
        return (7, [four, kicker])
    if by_count[0][1] == 3 and by_count[1][1] >= 2:
        # 葫芦: [三条点, 对子点]
        trips = by_count[0][0]
        pair = by_count[1][0]
        return (6, [trips, pair])
    if is_flush:
        # 同花: 5 张从大到小
        flush_ranks = sorted([c.rank for c in cards if c.suit == flush_suit], reverse=True)[:5]
        return (5, flush_ranks)
    st_high = straight_high(ranks)
    if st_high is not None:
        return (4, [st_high])  # 顺子

    if by_count[0][1] == 3:
        # 三条: [三条点, 两个踢脚]
        trips = by_count[0][0]
        kickers = [r for r in ranks if r != trips][:2]
        return (3, [trips] + kickers)
    if by_count[0][1] == 2 and by_count[1][1] == 2:
        # 两对: [高对, 低对, 踢脚]
        high_pair, low_pair = sorted([by_count[0][0], by_count[1][0]], reverse=True)
        kicker = max([r for r in ranks if r != high_pair and r != low_pair])
        return (2, [high_pair, low_pair, kicker])
    if by_count[0][1] == 2:
        # 一对: [对子点, 三个踢脚]
        pair = by_count[0][0]
        kickers = [r for r in ranks if r != pair][:3]
        return (1, [pair] + kickers)
    # 高牌: 5 张从大到小
    return (0, ranks[:5])

def best_7_of_5(cards7: List[Card]) -> Tuple[int, List[int], List[Card]]:
    # 7 中取 5，找最大
    best_cat = -1
    best_tie: List[int] = []
    best_hand: List[Card] = []
    for comb in itertools.combinations(cards7, 5):
        cat, tie = evaluate_5(list(comb))
        if cat > best_cat or (cat == best_cat and tie > best_tie):
            best_cat = cat
            best_tie = tie
            best_hand = list(comb)
    return best_cat, best_tie, best_hand

def category_name(cat: int) -> str:
    return ["High Card", "One Pair", "Two Pair", "Trips", "Straight",
            "Flush", "Full House", "Four of a Kind", "Straight Flush"][cat]

# =========================
# 下注轮与整手流程
# =========================
class BettingState:
    def __init__(self, p_btn: Player, p_bb: Player):
        self.players = [p_btn, p_bb]   # 0=BTN(SB), 1=BB
        self.pot = 0
        self.current_bet = 0           # 本轮单人最高投入
        self.invested = [0, 0]         # 本轮每人已投入
        self.last_raiser: Optional[int] = None
        self.round_over = False

    def reset_round(self):
        self.current_bet = 0
        self.invested = [0, 0]
        self.last_raiser = None
        for p in self.players:
            p.to_call = 0

    def reset_round(self):
        self.current_bet = 0
        self.last_raiser = None
        for p in self.players:
            p.to_call = 0

    def add_to_pot(self, amount: int):
        self.pot += amount
def recompute_to_calls(state: BettingState):
    for i, p in enumerate(state.players):
        need = state.current_bet - state.invested[i]
        p.to_call = max(0, need)
def post_blinds(state: BettingState):
    btn, bb = state.players
    sb_amt = min(btn.stack, SMALL_BLIND)
    bb_amt = min(bb.stack, BIG_BLIND)

    btn.stack -= sb_amt
    bb.stack  -= bb_amt
    state.add_to_pot(sb_amt + bb_amt)

    state.invested[0] = sb_amt          # ★ 记录本轮投入
    state.invested[1] = bb_amt
    state.current_bet = bb_amt          # ★ 本轮最高投入=BB
    recompute_to_calls(state)           # ★ 推导双方 to_call

    if btn.stack == 0: btn.all_in = True
    if bb.stack  == 0: bb.all_in  = True
    print(f"盲注：{btn.name} 投入 SB {sb_amt}，{bb.name} 投入 BB {bb_amt}。底池={state.pot}")


def parse_action(raw: str) -> Tuple[str, Optional[int]]:
    s = raw.strip().lower()
    if s == "fold":
        return ("fold", None)
    if s == "check":
        return ("check", None)
    if s == "call":
        return ("call", None)
    if s == "all-in" or s == "allin":
        return ("all-in", None)
    if s.startswith("bet "):
        try:
            x = int(s.split()[1])
            return ("bet", x)
        except:
            pass
    if s.startswith("raise "):
        try:
            x = int(s.split()[1])
            return ("raise", x)
        except:
            pass
    return ("invalid", None)

def prompt_action(player: Player, state: BettingState, street_name: str) -> Tuple[str, Optional[int]]:
    while True:
        base = f"[{street_name}] {player.name}（栈={player.stack}，需跟={player.to_call}，底池={state.pot}）输入动作："
        act = input(base).strip()
        action, amt = parse_action(act)
        if action != "invalid":
            return action, amt
        print("无效指令。可用：fold / check / call / bet X / raise X / all-in")

def apply_action(player_idx: int, action: str, amt: Optional[int],
                 state: BettingState, street: str) -> str:
    """
    返回: "invalid"（不切换），"continue"（切换对手），"end"（本轮结束）
    """
    p   = state.players[player_idx]
    opp = state.players[1 - player_idx]

    if p.all_in or p.has_folded:
        return "continue"

    # ---- FOLD ----
    if action == "fold":
        p.has_folded = True
        print(f"{p.name} 选择 FOLD，{opp.name} 直接赢下底池 {state.pot}。")
        return "end"

    # ---- CHECK ----
    if action == "check":
        if p.to_call > 0:
            print("你不能 check（当前有待跟金额）。")
            return "invalid"
        print(f"{p.name} 选择 CHECK。")
        return "continue"  # 结束由外层“双人都在无加注下行动过”判定

    # ---- CALL ----
    if action == "call":
        if p.to_call == 0:
            print("无需跟注（你可以 check）。")
            return "invalid"
        pay = min(p.stack, p.to_call)
        p.stack -= pay
        state.add_to_pot(pay)
        state.invested[player_idx] += pay
        if p.stack == 0:
            p.all_in = True
        print(f"{p.name} 选择 CALL {pay}。剩余栈={p.stack}。")

        recompute_to_calls(state)

        # 双方已持平
        if state.players[0].to_call == 0 and state.players[1].to_call == 0:
            if state.last_raiser is not None:
                state.last_raiser = None
                return "end"       # 下注/加注被跟住 → 本轮结束
            if street == "Preflop" and player_idx == 0:
                return "continue"  # 翻前 SB 补到 BB → 让 BB 行动
        return "continue"

    # ---- ALL-IN ----
    if action == "all-in":
        if p.stack <= 0 and p.to_call == 0:
            print("无可投入筹码。")
            return "invalid"
        need = max(0, state.current_bet - state.invested[player_idx])
        call_pay = min(need, p.stack)
        p.stack -= call_pay
        state.add_to_pot(call_pay)
        state.invested[player_idx] += call_pay

        push = p.stack
        p.stack = 0
        state.add_to_pot(push)
        state.invested[player_idx] += push
        p.all_in = True

        if state.invested[player_idx] > state.current_bet:
            state.current_bet = state.invested[player_idx]
            state.last_raiser = player_idx

        recompute_to_calls(state)
        print(f"{p.name} 选择 ALL-IN！当前底池={state.pot}。")
        return "continue"

    # ---- BET ----（仅翻后且当前无人下注）
    if action == "bet":
        if street == "Preflop" or state.current_bet > 0:
            print("当前已有人下注（或翻前有大盲），请用 raise。")
            return "invalid"
        if amt is None or amt <= 0 or amt > p.stack:
            print("bet 金额非法。")
            return "invalid"
        p.stack -= amt
        state.add_to_pot(amt)
        state.invested[player_idx] += amt
        state.current_bet = state.invested[player_idx]
        state.last_raiser = player_idx
        recompute_to_calls(state)
        print(f"{p.name} 下注 {amt}。底池={state.pot}。")
        return "continue"

    # ---- RAISE ----（把“本轮你的总投入”提高到 amt）
    if action == "raise":
        if state.current_bet == 0:
            print("当前无人下注，不能 raise，请使用 bet。")
            return "invalid"
        if amt is None or amt <= state.current_bet:
            print(f"raise 目标必须大于当前注额 {state.current_bet}。")
            return "invalid"
        target = amt
        need_total = target - state.invested[player_idx]
        if need_total <= 0:
            print("你的投入已达到/超过该目标。")
            return "invalid"
        if need_total > p.stack:
            print("你的栈不足以加到该额度。")
            return "invalid"

        p.stack -= need_total
        state.add_to_pot(need_total)
        state.invested[player_idx] += need_total
        state.current_bet = target
        state.last_raiser = player_idx
        recompute_to_calls(state)
        print(f"{p.name} 加注到 {state.current_bet}。底池={state.pot}。")
        return "continue"

    return "invalid"



def run_betting_round(state: BettingState, street: str, first_to_act_idx: int):
    if street != "Preflop":
        state.reset_round()

    idx = first_to_act_idx
    acted_after_raise = set()

    while True:
        p0, p1 = state.players
        # 提前结束：有人弃牌 / 双方全下
        if p0.has_folded or p1.has_folded:
            state.round_over = True
            return
        if (p0.all_in or p0.stack == 0) and (p1.all_in or p1.stack == 0):
            state.round_over = True
            return

        player = state.players[idx]
        opponent = state.players[1 - idx]

        if player.all_in or player.has_folded:
            idx = 1 - idx
            continue

        # --- 提示动作 ---
        action, amt = prompt_action(player, state, street)
        before_raiser = state.last_raiser
        result = apply_action(idx, action, amt, state, street)

        if result == "invalid":
            # 不切换，继续让当前玩家输入
            continue

        if result == "end":
            state.round_over = True
            return

        # result == "continue"
        if state.last_raiser is not None and state.last_raiser != before_raiser:
            acted_after_raise = {idx}  # 有新加注 → 重置集合
        else:
            acted_after_raise.add(idx)
            if (state.players[0].to_call == 0 and state.players[1].to_call == 0
                and state.last_raiser is None and len(acted_after_raise) == 2):
                state.round_over = True
                return

        # ★ 切换到对手
        idx = 1 - idx

def showdown(p1: Player, p2: Player, board: List[Card], pot: int):
    print("\n=== 摊牌 SHOWDOWN ===")
    print(f"{p1.name} 手牌: {p1.hole}")
    print(f"{p2.name} 手牌: {p2.hole}")
    print(f"公牌: {board}")

    cat1, tie1, best5_1 = best_7_of_5(p1.hole + board)
    cat2, tie2, best5_2 = best_7_of_5(p2.hole + board)

    print(f"{p1.name}: {category_name(cat1)} | 手牌组合: {best5_1} | 关键值: {tie1}")
    print(f"{p2.name}: {category_name(cat2)} | 手牌组合: {best5_2} | 关键值: {tie2}")

    if cat1 > cat2 or (cat1 == cat2 and tie1 > tie2):
        print(f"胜者：{p1.name} 赢得 {pot}")
        p1.stack += pot
    elif cat2 > cat1 or (cat1 == cat2 and tie2 > tie1):
        print(f"胜者：{p2.name} 赢得 {pot}")
        p2.stack += pot
    else:
        print("平分底池！")
        split = pot // 2
        p1.stack += split
        p2.stack += pot - split  # 奇数分配
import pdb
def play_one_hand(p_btn: Player, p_bb: Player) -> None:
    """进行一手牌；按钮=小盲，翻前按钮先行动；翻后大盲先行动"""
    # 准备
    for p in (p_btn, p_bb):
        p.reset_for_new_hand()
    deck = Deck()
    deck.shuffle()
    state = BettingState(p_btn, p_bb)

    # 发底牌
    p_btn.hole = deck.deal(2)
    p_bb.hole = deck.deal(2)

    print("\n==============================")
    print(f"按钮位(小盲)：{p_btn.name}  | 大盲：{p_bb.name}")
    print(f"{p_btn.name} 手牌: {p_btn.hole}")
    # 你可以隐藏对手手牌显示：这里只为测试演示
    print(f"{p_bb.name} 手牌: {p_bb.hole}")

    # 盲注
    post_blinds(state)

    # 翻前：按钮先行动（Heads-Up 规则）
    run_betting_round(state, "Preflop", first_to_act_idx=0)
    if state.players[0].has_folded or state.players[1].has_folded:
        return

    # 翻牌
    deck.burn()
    board = deck.deal(3)
    print(f"\n翻牌 FLOP: {board}")
    run_betting_round(state, "Flop", first_to_act_idx=1)  # 翻后大盲先行动
    if state.players[0].has_folded or state.players[1].has_folded:
        return

    # 转牌
    deck.burn()
    board += deck.deal(1)
    print(f"\n转牌 TURN: {board}")
    run_betting_round(state, "Turn", first_to_act_idx=1)
    if state.players[0].has_folded or state.players[1].has_folded:
        return

    # 河牌
    deck.burn()
    board += deck.deal(1)
    print(f"\n河牌 RIVER: {board}")
    run_betting_round(state, "River", first_to_act_idx=1)
    if state.players[0].has_folded or state.players[1].has_folded:
        return

    # 摊牌
    showdown(state.players[0], state.players[1], board, state.pot)

def main():
    print("=== 两人德州扑克（Heads-Up） ===")
    name1 = input("玩家1 名字（按钮位先手）：").strip() or "Player1"
    name2 = input("玩家2 名字（大盲位）：").strip() or "Player2"
    p1 = Player(name1, STARTING_STACK)
    p2 = Player(name2, STARTING_STACK)
    # 初始按钮位：p1
    p1.is_button = True
    p2.is_button = False
    hand_no = 1
    while True:
        # 按钮位与大盲位分配
        btn = p1 if p1.is_button else p2
        bb = p2 if p1.is_button else p1
        print(f"\n--- 第 {hand_no} 手 | {btn.name}(BTN/SB) vs {bb.name}(BB) ---")
        print(f"筹码：{p1.name}={p1.stack}，{p2.name}={p2.stack}")
        if p1.stack <= 0 or p2.stack <= 0:
            print("有玩家破产。游戏结束。")
            break
        play_one_hand(btn, bb)
        print(f"\n手牌结束。筹码：{p1.name}={p1.stack}，{p2.name}={p2.stack}")
        cont = input("回车继续下一手；输入 q 退出：").strip().lower()
        if cont == "q":
            break
        # 轮换按钮
        p1.is_button, p2.is_button = p2.is_button, p1.is_button
        hand_no += 1

if __name__ == "__main__":
    main()