"""
我和调用的API玩德扑
note: 记录n个玩家的下注
提供下注端口
"""
import random
import itertools
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from api import OpenAIClient
import argparse
import re, unicodedata, copy, hashlib
from models import HandLog, Street, ActionEntry, ParsedDecision
from recorder import record_step, make_observation

# =========================
# 配置（可自行修改）
# =========================
SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 5000
ALLOW_INVALID_INPUT_RETRY = True  # 输入错了反复提示
# 可供随机挑选的名字池 (2-6 人独一无二)
NAME_POOL = ["Ethan", "Sophia", "Liam", "Olivia", "Mason", "Ava"]

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
    def __init__(self, name: str, stack: int, args):
        self.name = name
        self.agent = OpenAIClient(**args)
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
    """支持 2-6 人的下注状态。原先只支持两人，这里推广。

    players: 顺时针顺序，button_index 指向按钮位。
    每轮只维护一个 main pot（未实现 side pot；多 All-in 时不完全公平——已在注释声明）。
    """
    def __init__(self, players: List[Player], button_index: int, hand_log: Optional[HandLog] = None):
        self.players: List[Player] = players
        self.button_index = button_index
        self.pot = 0
        self.current_bet = 0               # 本轮单人最高投入
        # invested 由 list 改为 dict：{player_name: 已在本轮投入}
        self.invested: Dict[str, int] = {p.name: 0 for p in players}
        self.last_raiser: Optional[int] = None
        self.round_over = False
        self.hand_log = hand_log

    @property
    def alive_indices(self) -> List[int]:
        return [i for i,p in enumerate(self.players) if (not p.has_folded) and (p.stack>0 or not p.all_in)]

    def reset_round(self):
        self.current_bet = 0
        self.last_raiser = None
        self.invested = {p.name: 0 for p in self.players}
        for p in self.players:
            p.to_call = 0

    def add_to_pot(self, amount: int):
        self.pot += amount

    # 兼容 recorder.record_step 里调用
    def apply_action_by_parsed(self, actor_name: str, action: str, amt: Optional[int]):
        try:
            idx = next(i for i,p in enumerate(self.players) if p.name == actor_name)
        except StopIteration:
            return
        # 调用通用 apply_action_multi（不改变轮次结束逻辑，这里只做状态更新）
        apply_action(idx, action, amt, self, "(replay)")
def recompute_to_calls(state: BettingState):
    for p in state.players:
        need = state.current_bet - state.invested[p.name]
        p.to_call = max(0, need)

def post_blinds(state: BettingState, street_name: str = "Preflop"):
    n = len(state.players)
    if n == 2:
        # 传统 HU：按钮即小盲，对手大盲
        btn_idx = state.button_index
        sb_idx = btn_idx
        bb_idx = 1 - btn_idx
    else:
        btn_idx = state.button_index
        sb_idx = (btn_idx + 1) % n
        bb_idx = (btn_idx + 2) % n

    sb_p = state.players[sb_idx]
    bb_p = state.players[bb_idx]
    sb_amt = min(sb_p.stack, SMALL_BLIND)
    bb_amt = min(bb_p.stack, BIG_BLIND)
    prev_inv_sb = state.invested[sb_p.name]
    prev_inv_bb = state.invested[bb_p.name]

    sb_p.stack -= sb_amt
    bb_p.stack -= bb_amt
    state.invested[sb_p.name] += sb_amt
    state.invested[bb_p.name] += bb_amt
    state.add_to_pot(sb_amt + bb_amt)
    state.current_bet = max(state.invested.values())
    recompute_to_calls(state)

    if state.hand_log:
        state.hand_log.add_action(ActionEntry(
            street=street_name, actor=sb_p.name, action="SB", amount=sb_amt,
            to_call_before=0, invested_before=prev_inv_sb,
            invested_after=state.invested[sb_p.name], pot_after=state.pot, note=""
        ))
        state.hand_log.add_action(ActionEntry(
            street=street_name, actor=bb_p.name, action="BB", amount=bb_amt,
            to_call_before=0, invested_before=prev_inv_bb,
            invested_after=state.invested[bb_p.name], pot_after=state.pot, note=""
        ))

    if sb_p.stack == 0: sb_p.all_in = True
    if bb_p.stack == 0: bb_p.all_in = True
    print(f"盲注：{sb_p.name} 投入 SB {sb_amt}，{bb_p.name} 投入 BB {bb_amt}。底池={state.pot}")



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
_DECISION_LINE_RE = re.compile(
    r"""
    (?imx)                              # 多行 / 忽略大小写 / 可注释
    ^\s*decision\s*[:：]\s*             # 'decision:'（容忍空格/中文冒号）
    (?P<raw>                            # <== 关键：把整个动作当成一坨抓出来
        (?:
            (?P<fold>fold) |
            (?P<check>check) |
            (?P<call>call) |
            all[-\s]?in(?:\s+for\s+(?P<allin_amt>\d+))? |
            bet\s+(?P<bet>\d+) |
            raise\s+(?:to\s+)?(?P<raise_to>\d+)
        )
    )
    \s*$                                # 行尾
    """,
)
def parse_last_decision_line(response_text: str) -> ParsedDecision:
    t = unicodedata.normalize("NFKC", response_text)
    matches = list(_DECISION_LINE_RE.finditer(t))
    if not matches:
        raise ValueError("No valid 'decision:' line found.")
    m = matches[-1]
    raw = m.group("raw")
    if m.group("fold"):  return ParsedDecision("fold", text=raw)
    if m.group("check"): return ParsedDecision("check", text=raw)
    if m.group("call"):  return ParsedDecision("call", text=raw)
    if m.group("bet") is not None:
        amt = int(m.group("bet"))
        return ParsedDecision("bet", amount=amt, text=raw)
    if m.group("raise_to") is not None:
        to  = int(m.group("raise_to"))
        return ParsedDecision("raise", to=to, text=raw)
    # all-in
    amt = int(m.group("allin_amt")) if m.group("allin_amt") else None
    return ParsedDecision("all-in", amount=amt, text=raw) 
def prompt_action(player, state, street_name: str):
    # 1) 组 observation（下一行会内嵌你已有的 History/Board/等）
    obs = make_observation(state, player.name)
    prompt_text = obs.to_prompt_text()

    # 2) 调模型（保留完整 response_text 以便“思考过程”存档）
    # pdb.set_trace()
    resp_text, _, cost = player.agent(model_name='gpt-4o-mini', prompt=prompt_text)  # 你的返回是 (text, _, cost)

    # 3) 解析最后一条 decision 行
    decision = parse_last_decision_line(resp_text)

    # 4) 记录一步（含应用前后状态）

    record_step(
        hand_log=state.hand_log,
        state=state,
        actor_name=player.name,
        model_name='gpt-4o-mini',
        response_text=resp_text,
        decision=decision,
        obs=obs
    )

    # 5) 返回给你的原有引擎（如果你不在 record_step 里 apply，就在这里 apply 再返回）
    if decision.action in ("bet",):
        return "bet", decision.amount
    if decision.action in ("raise",):
        return "raise", decision.to
    return decision.action, None

def apply_action(player_idx: int, action: str, amt: Optional[int],
                 state: BettingState, street: str) -> str:
    """
    返回: "invalid"（不切换），"continue"（切换对手），"end"（本轮结束）
    """
    p   = state.players[player_idx]
    # 找一个存活对手用于打印（不严格指定）
    opp = None
    for j,other in enumerate(state.players):
        if j != player_idx and not other.has_folded:
            opp = other
            break
    prev_invested = state.invested[p.name]
    prev_to_call = p.to_call
    if p.all_in or p.has_folded:
        return "continue"
    # ---- FOLD ----
    if action == "fold":
        p.has_folded = True
        alive_after = [pl for pl in state.players if (not pl.has_folded)]
        if opp:
            print(f"{p.name} 选择 FOLD。")
        # 记历史
        if state.hand_log:
            state.hand_log.add_action(ActionEntry(
                street=street, actor=p.name, action="fold", amount=0,
                to_call_before=prev_to_call, invested_before=prev_invested,
                invested_after=state.invested[p.name], pot_after=state.pot
            ))
        if len(alive_after) == 1:
            # 直接结束整手牌
            winner = alive_after[0]
            print(f"{winner.name} 因对手全部弃牌赢下底池 {state.pot}。")
            winner.stack += state.pot
            state.pot = 0
            return "hand_end"
        return "continue"

    # ---- CHECK ----
    if action == "check":
        if p.to_call > 0:
            print("你不能 check（当前有待跟金额）。")
            return "invalid"
        print(f"{p.name} 选择 CHECK。")
        if state.hand_log:
            state.hand_log.add_action(ActionEntry(
                street=street, actor=p.name, action="check", amount=0,
                to_call_before=prev_to_call, invested_before=prev_invested,
                invested_after=state.invested[p.name], pot_after=state.pot
            ))
        return "continue"  # 结束由外层“双人都在无加注下行动过”判定

    # ---- CALL ----
    if action == "call":
        if p.to_call == 0:
            print("无需跟注（你可以 check）。")
            return "invalid"
        pay = min(p.stack, p.to_call)
        p.stack -= pay
        state.add_to_pot(pay)
        state.invested[p.name] += pay
        if p.stack == 0:
            p.all_in = True
        print(f"{p.name} 选择 CALL {pay}。剩余栈={p.stack}。")

        recompute_to_calls(state)
        # 多人：整轮结束判定由外层处理
        if state.hand_log:
            amount_delta = state.invested[p.name] - prev_invested
            state.hand_log.add_action(ActionEntry(
                street=street, actor=p.name, action="call", amount=amount_delta,
                to_call_before=prev_to_call, invested_before=prev_invested,
                invested_after=state.invested[p.name], pot_after=state.pot,
                note=("preflop SB completes to BB" if (street=="Preflop" and prev_to_call>0 and state.last_raiser is None) else "")
            ))            
        return "continue"

    # ---- ALL-IN ----
    if action == "all-in":
        if p.stack <= 0 and p.to_call == 0:
            print("无可投入筹码。")
            return "invalid"
        need = max(0, state.current_bet - state.invested[p.name])
        call_pay = min(need, p.stack)
        p.stack -= call_pay
        state.add_to_pot(call_pay)
        state.invested[p.name] += call_pay

        push = p.stack
        p.stack = 0
        state.add_to_pot(push)
        state.invested[p.name] += push
        p.all_in = True

        if state.invested[p.name] > state.current_bet:
            state.current_bet = state.invested[p.name]
            state.last_raiser = player_idx

        recompute_to_calls(state)
        print(f"{p.name} 选择 ALL-IN！当前底池={state.pot}。")
        if state.hand_log:
            amount_delta = state.invested[p.name] - prev_invested
            state.hand_log.add_action(ActionEntry(
                street=street, actor=p.name, action="all-in", amount=amount_delta,
                to_call_before=prev_to_call, invested_before=prev_invested,
                invested_after=state.invested[p.name], pot_after=state.pot
            ))
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
        state.invested[p.name] += amt
        state.current_bet = state.invested[p.name]
        state.last_raiser = player_idx
        recompute_to_calls(state)
        print(f"{p.name} 下注 {amt}。底池={state.pot}。")
        if state.hand_log:
            amount_delta = state.invested[p.name] - prev_invested
            state.hand_log.add_action(ActionEntry(
                street=street, actor=p.name, action="bet", amount=amount_delta,
                to_call_before=prev_to_call, invested_before=prev_invested,
                invested_after=state.invested[p.name], pot_after=state.pot
            ))        
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
        need_total = target - state.invested[p.name]
        if need_total <= 0:
            print("你的投入已达到/超过该目标。")
            return "invalid"
        if need_total > p.stack:
            print("你的栈不足以加到该额度。")
            return "invalid"

        p.stack -= need_total
        state.add_to_pot(need_total)
        state.invested[p.name] += need_total
        state.current_bet = target
        state.last_raiser = player_idx
        recompute_to_calls(state)
        print(f"{p.name} 加注到 {state.current_bet}。底池={state.pot}。")
        if state.hand_log:
            amount_delta = state.invested[p.name] - prev_invested
            state.hand_log.add_action(ActionEntry(
                street=street, actor=p.name, action="raise", amount=amount_delta,
                to_call_before=prev_to_call, invested_before=prev_invested,
                invested_after=state.invested[p.name], pot_after=state.pot
            ))        
        return "continue"

    return "invalid"



def run_betting_round(state: BettingState, street: str, first_to_act_idx: int) -> None:
    if street != "Preflop":
        state.reset_round()

    n = len(state.players)
    acted_after_raise = set()
    idx = first_to_act_idx

    # 跳过已弃牌 / 全下玩家
    def next_index(i):
        for step in range(1, n+1):
            ni = (i + step) % n
            p = state.players[ni]
            if not p.has_folded and not (p.all_in or p.stack == 0 and state.invested[p.name] == state.current_bet):
                return ni
        return i

    while True:
        active_players = [p for p in state.players if not p.has_folded]
        if len(active_players) <= 1:
            state.round_over = True
            return
        # 如果所有非 all-in 玩家都已 all-in，则直接结束
        if all(p.all_in or p.stack == 0 for p in active_players):
            state.round_over = True
            return

        player = state.players[idx]
        if player.has_folded or (player.all_in or player.stack == 0):
            idx = next_index(idx)
            continue

        action, amt = prompt_action(player, state, street)
        before_raiser = state.last_raiser
        result = apply_action(idx, action, amt, state, street)

        if result == "invalid":
            continue
        if result == "hand_end":  # 整手牌已有人获胜
            state.round_over = True
            return

        # 下注轮是否可结束：所有存活玩家 to_call ==0 且 无 last_raiser 且 每人自上次加注后至少行动一次
        if state.last_raiser is not None and state.last_raiser != before_raiser:
            acted_after_raise = {idx}
        else:
            acted_after_raise.add(idx)
            # 参与者集合（未弃牌且未全下前投入差额）
            need_act_indices = [i for i,p in enumerate(state.players)
                                if (not p.has_folded) and not p.all_in]
            if all(state.players[i].to_call == 0 for i in need_act_indices) and \
               state.last_raiser is None and \
               acted_after_raise.issuperset(need_act_indices):
                state.round_over = True
                return

        idx = next_index(idx)

def showdown_multi(players: List[Player], board: List[Card], pot: int):
    print("\n=== 摊牌 SHOWDOWN ===")
    print(f"公牌: {board}")
    evals = []
    for p in players:
        if p.has_folded:
            continue
        print(f"{p.name} 手牌: {p.hole}")
        cat, tie, best5 = best_7_of_5(p.hole + board)
        print(f"  -> {category_name(cat)} {best5} 关键: {tie}")
        evals.append((cat, tie, p))
    if not evals:
        print("无人参与摊牌。")
        return
    # 按 cat / tie 排序取最高
    evals.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_cat, best_tie, _ = evals[0]
    winners = [p for c,t,p in evals if c==best_cat and t==best_tie]
    if len(winners) == 1:
        w = winners[0]
        print(f"胜者：{w.name} 赢得 {pot}")
        w.stack += pot
    else:
        print(f"平分底池：{' / '.join(p.name for p in winners)}")
        share = pot // len(winners)
        remainder = pot - share*len(winners)
        for i,p in enumerate(winners):
            p.stack += share + (1 if i < remainder else 0)
import os, datetime
def play_one_hand_multi(players: List[Player], button_index: int, hand_no: int = 1, save_dir: str = "logs") -> int:
    """进行一手牌。返回新的按钮 index (下一手使用)。未实现 side pot。"""
    # 重置
    for p in players:
        p.reset_for_new_hand()
    deck = Deck(); deck.shuffle()
    # 简化 HandLog：只保存按钮名（构造仍沿用原 API 占位两玩家，不严格记录多玩家）
    dummy_btn = players[button_index]
    dummy_bb  = players[(button_index+1)%len(players)] if len(players)>1 else players[0]
    hand_log = HandLog(hand_no, button_name=dummy_btn.name, sb=SMALL_BLIND, bb=BIG_BLIND,
                       p_btn=dummy_btn, p_bb=dummy_bb, players=players)

    # 发底牌
    for p in players:
        p.hole = deck.deal(2)
        hand_log.set_hole_cards(p.name, p.hole)

    state = BettingState(players, button_index, hand_log=hand_log)
    print("\n==============================")
    order_str = " -> ".join(p.name for p in players)
    print(f"手 {hand_no} | 座位顺序(按钮*标记)：{order_str}")
    for i,p in enumerate(players):
        tag = "(BTN)" if i==button_index else ""
        print(f"{p.name}{tag} 手牌: {p.hole}")

    # 盲注
    post_blinds(state, "Preflop")

    # 翻前首个行动者
    if len(players) == 2:
        first_idx = state.button_index  # HU 按钮先
    else:
        first_idx = (button_index + 3) % len(players) if len(players) >=3 else (button_index+1)%len(players)  # UTG
    run_betting_round(state, "Preflop", first_idx)
    # 若已有人赢下整手（fold 清光）则直接结束
    if state.pot == 0 or sum(0 if p.has_folded else 1 for p in players) <=1:
        return (button_index + 1) % len(players)

    # 翻牌
    board: List[Card] = []
    deck.burn(); board = deck.deal(3); hand_log.set_board("Flop", board); print(f"\n翻牌 FLOP: {board}")
    first_flop = (button_index + 1) % len(players) if len(players)>2 else (state.button_index ^ 1)
    run_betting_round(state, "Flop", first_flop)
    if state.pot == 0 or sum(0 if p.has_folded else 1 for p in players) <=1:
        return (button_index + 1) % len(players)

    # 转牌
    deck.burn(); board += deck.deal(1); hand_log.set_board("Turn", board); print(f"\n转牌 TURN: {board}")
    run_betting_round(state, "Turn", first_flop)
    if state.pot == 0 or sum(0 if p.has_folded else 1 for p in players) <=1:
        return (button_index + 1) % len(players)

    # 河牌
    deck.burn(); board += deck.deal(1); hand_log.set_board("River", board); print(f"\n河牌 RIVER: {board}")
    run_betting_round(state, "River", first_flop)
    if state.pot == 0 or sum(0 if p.has_folded else 1 for p in players) <=1:
        return (button_index + 1) % len(players)

    # 摊牌
    remaining = [p for p in players if not p.has_folded]
    showdown_multi(remaining, board, state.pot)
    state.pot = 0
    # 保存日志
    try:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        path = os.path.join(save_dir, f"hand_{hand_no}_{ts}.json")
        hand_log.save_json(path)
    except Exception as e:
        print(f"保存手牌日志失败: {e}")
    return (button_index + 1) % len(players)
def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    ################ args ################
    parser.add_argument('--api_batch_use', type=bool, default=False, help='wheather use batch api.')

    args, unparsed = parser.parse_known_args()
    return args
def main():
    print("=== N 人德州扑克 (2-6) ===")
    while True:
        try:
            n = int(input("请输入玩家数量 (2-6)：").strip())
        except ValueError:
            print("请输入数字。")
            continue
        if 2 <= n <= 6:
            break
        print("范围错误。")

    args = parse_option()
    api_args = {"temperature":0.9, "max_tokens":1024, "args": args}

    # 随机抽取不重复名字
    chosen_names = random.sample(NAME_POOL, n)
    players: List[Player] = [Player(name, STARTING_STACK, api_args) for name in chosen_names]
    print("座位 (顺时针)：", " -> ".join(p.name for p in players))

    button_index = 0  # 初始按钮设为列表第 0 个
    hand_no = 1
    while True:
        # 检查是否仍然有至少两个玩家有筹码
        alive_with_stack = [p for p in players if p.stack > 0]
        if len(alive_with_stack) < 2:
            print("游戏结束：只剩一名玩家有筹码。")
            break
        print(f"\n--- 第 {hand_no} 手 ---")
        print("筹码:", ", ".join(f"{p.name}={p.stack}" for p in players))
        button_index = play_one_hand_multi(players, button_index, hand_no, save_dir="logs")
        print("\n手牌结束后筹码:", ", ".join(f"{p.name}={p.stack}" for p in players))
        cont = input("回车继续；输入 q 退出：").strip().lower()
        if cont == 'q':
            break
        hand_no += 1
if __name__ == "__main__":
    main()