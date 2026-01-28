"""
我和调用的API玩德扑
note: 记录n个玩家的下注
提供下注端口
"""
import random
import itertools
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from api import OpenAIClient
import argparse, os, json, glob
import re, unicodedata, copy, hashlib
from models import HandLog, Street, ActionEntry, ParsedDecision
from recorder import record_step, make_observation, cost_record

# =========================
# 配置（可自行修改）
# =========================
SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 5000
ALLOW_INVALID_INPUT_RETRY = True  # 输入错了反复提示
# 可供随机挑选的名字池 (2-6 人独一无二)
NAME_POOL = ["Ethan", "Sophia", "Liam", "Olivia", "Mason", "Ava"]

FIX_MODE_PRESET = {
    "button": {"name": "Johnny Chan", "hole": ["Qc", "7h"]},
    "bigblind": {"name": "Erik Seidel", "hole": ["Jc", "9c"]},
    "board": ["Qc", "8d", "Th", "2s", "6d"]   # Flop(3) + Turn + River
}

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
        # 策略总结（跨手持久）与版本历史
        self.strategy_summary: str = ""
        self.strategy_versions: List[Dict[str, str]] = []  # {hand_no, text}
        self.enable_summary: bool = False

    def reset_for_new_hand(self):
        self.hole = []
        self.to_call = 0
        self.all_in = False
        self.has_folded = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stack": self.stack,
            "hole": [asdict(c) for c in self.hole],
            "is_button": self.is_button,
            "to_call": self.to_call,
            "all_in": self.all_in,
            "has_folded": self.has_folded,
            "strategy_summary": self.strategy_summary,
            "strategy_versions": self.strategy_versions,
            "enable_summary": self.enable_summary,
            "steps": [
                asdict(s) if hasattr(s, "__dataclass_fields__") else s
                for s in self.steps
            ],
        }

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
    def __init__(self, players: List[Player], button_index: int,
                 hand_log: Optional[HandLog] = None,
                 game_mode: str = "random"):
        self.players: List[Player] = players
        self.button_index = button_index
        self.pot = 0
        self.current_bet = 0               # 本轮单人最高投入
        # invested 由 list 改为 dict：{player_name: 已在本轮投入}
        self.invested: Dict[str, int] = {p.name: 0 for p in players}
        self.last_raiser: Optional[int] = None
        self.round_over = False
        self.hand_log = hand_log
        self.game_mode = game_mode

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
            idx = next(i for i in range(len(self.players)) if self.players[i].name == actor_name)
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
        # 1) Embedded pattern search anywhere in text (handles narrative like: Therefore ... 'decision: call'.)
        embedded_pat = re.compile(
            r"(?im)decision\s*[:：]\s*(fold|check|call|all[-\s]?in(?:\s+for\s+\d+)?|raise(?:\s+to)?\s+\d+)"
        )
        embedded = list(embedded_pat.finditer(t))
        if embedded:
            raw = embedded[-1].group(1)
            raw_norm = raw.lower().strip().strip("'\". ")
            # Normalize 'raise to' vs 'raise'
            if raw_norm in ("fold","check","call"):
                return ParsedDecision(raw_norm, text=raw)
            m_raise = re.match(r"^raise(?:\s+to)?\s+(\d+)$", raw_norm)
            if m_raise:
                return ParsedDecision("raise", to=int(m_raise.group(1)), text=raw)
            m_allin = re.match(r"^all[-\s]?in(?:\s+for\s+(\d+))?$", raw_norm)
            if m_allin:
                amt = m_allin.group(1)
                return ParsedDecision("all-in", amount=int(amt) if amt else None, text=raw)
        # 2) Heuristic fallback: inspect last non-empty lines (still supporting pure action words)
        lines = [ln.strip().strip("'\"") for ln in t.splitlines() if ln.strip()]
        for raw_line in reversed(lines[-10:]):  # limit search window
            low = raw_line.lower().strip("'\". ")
            if low in ("fold","check","call"):
                return ParsedDecision(low, text=raw_line)
            m_raise = re.match(r"^raise(?:\s+to)?\s+(\d+)$", low)
            if m_raise:
                return ParsedDecision("raise", to=int(m_raise.group(1)), text=raw_line)
            m_allin = re.match(r"^all[-\s]?in(?:\s+for\s+(\d+))?$", low)
            if m_allin:
                amt = m_allin.group(1)
                return ParsedDecision("all-in", amount=int(amt) if amt else None, text=raw_line)
        # 3) If still not found, raise with preview
        preview = t[-400:] if len(t) > 400 else t
        raise ValueError("No valid 'decision:' line found (preview tail):\n" + preview)
    m = matches[-1]
    raw = m.group("raw")
    if m.group("fold"):  return ParsedDecision("fold", text=raw)
    if m.group("check"): return ParsedDecision("check", text=raw)
    if m.group("call"):  return ParsedDecision("call", text=raw)
    if m.group("raise_to") is not None:
        to  = int(m.group("raise_to"))
        return ParsedDecision("raise", to=to, text=raw)
    # all-in
    amt = int(m.group("allin_amt")) if m.group("allin_amt") else None
    return ParsedDecision("all-in", amount=amt, text=raw) 
import pdb
import time

def prompt_action(player, state, street_name: str):
    obs = make_observation(state, player.name)
    prompt_text = obs.to_prompt_text(state)
    max_retry = 3
    backoff = 1.0
    for attempt in range(max_retry):
        try:
            resp_text, _, cost = player.agent(model_name='gpt-4o-mini', prompt=prompt_text)
        except Exception as e:
            print(f"[APIError] attempt {attempt+1}/{max_retry}: {e}")
            time.sleep(backoff); backoff = min(backoff*2, 8.0)
            continue

        if cost:
            cost_record(cost)

        try:
            decision = parse_last_decision_line(resp_text)
        except Exception as e:
            print(f"[ParseError] attempt {attempt+1}/{max_retry}: {e}")
            time.sleep(backoff); backoff = min(backoff*2, 8.0)
            continue

        # 轻量合法性检查（可选）
        legal_names = set()
        for la in getattr(obs, "legal_actions", []) or []:
            if isinstance(la, str):
                legal_names.add(la.lower())
            elif isinstance(la, dict):
                legal_names.add(str(la.get("action", "")).lower())
            else:
                name = getattr(la, "action", getattr(la, "name", None))
                if name: legal_names.add(str(name).lower())

        act = decision.action.lower()
        if legal_names and act not in legal_names:
            print(f"[IllegalDecision] '{act}' not in legal_actions {sorted(legal_names)}; retrying...")
            time.sleep(backoff); backoff = min(backoff*2, 8.0)
            continue
        if act == "raise" and getattr(decision, "to", None) is None:
            print("[InvalidRaise] missing 'to' amount; retrying...")
            time.sleep(backoff); backoff = min(backoff*2, 8.0)
            continue

        record_step(
            hand_log=state.hand_log,
            state=state,
            actor_name=player.name,
            model_name='gpt-4o-mini',
            prompt_text=prompt_text,
            response_text=resp_text,
            decision=decision,
            obs=obs
        )
        if act == "raise":
            return "raise", decision.to
        return act, None

    raise ValueError(f"LLM连续{max_retry}次输出无法解析/非法，已放弃。")

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
            # 容错：模型输出 call 但无需跟注 -> 等价于 check
            print(f"{p.name} 输入 'call' 但无需跟注，视为 CHECK。")
            if state.hand_log:
                state.hand_log.add_action(ActionEntry(
                    street=street, actor=p.name, action="check", amount=0,
                    to_call_before=prev_to_call, invested_before=prev_invested,
                    invested_after=state.invested[p.name], pot_after=state.pot,
                    note="call-as-check"
                ))
            return "continue"
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

    # ---- RAISE ----（首次进攻也用 raise，表示把“本轮总投入”提高到 amt）
    if action == "raise":
        if amt is None or amt <= 0:
            print("raise 目标无效。")
            return "invalid"
        # 首次进攻：current_bet==0 时直接允许 raise to amt
        if state.current_bet > 0 and amt <= state.current_bet:
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
        verb = "加注" if state.current_bet > 0 else "首次加注"
        print(f"{p.name} {verb}到 {state.current_bet}。底池={state.pot}。")
        if state.hand_log:
            amount_delta = state.invested[p.name] - prev_invested
            state.hand_log.add_action(ActionEntry(
                street=street, actor=p.name, action="raise", amount=amount_delta,
                to_call_before=prev_to_call, invested_before=prev_invested,
                invested_after=state.invested[p.name], pot_after=state.pot
            ))
        return "continue"

    return "invalid"



def run_betting_round(state, street, first_to_act_idx):
    if street != "Preflop":
        state.reset_round()

    n = len(state.players)
    acted_after_raise = set()
    idx = first_to_act_idx

    def next_index(i):
        for step in range(1, n+1):
            ni = (i + step) % n
            p = state.players[ni]
            if not p.has_folded and not (p.all_in or (p.stack == 0 and state.invested[p.name] == state.current_bet)):
                return ni
        return i

    safety = 0
    while True:
        safety += 1
        if safety > 500:
            print(f"[WARN] Safety break on street {street}. last_raiser={state.last_raiser}")
            state.round_over = True
            return

        alive_players = [p for p in state.players if not p.has_folded]
        if len(alive_players) <= 1:
            state.round_over = True
            return

        # 如果仅剩 1 个未 all-in 且未弃牌玩家（其他都 all-in 或弃牌），无需继续动作
        need_active = [p for p in alive_players if not p.all_in]
        if len(need_active) <= 1:
            state.round_over = True
            return

        player = state.players[idx]
        if player.has_folded or player.all_in or player.stack == 0:
            idx = next_index(idx)
            continue

        action, amt = prompt_action(player, state, street)
        before_raiser = state.last_raiser
        result = apply_action(idx, action, amt, state, street)

        if result == "invalid":
            continue
        if result == "hand_end":
            state.round_over = True
            return

        # 下注轮结束判定
        if state.last_raiser is not None and state.last_raiser != before_raiser:
            acted_after_raise = {idx}
        else:
            acted_after_raise.add(idx)

        need_act_indices = [
            i for i, p in enumerate(state.players)
            if (not p.has_folded) and not p.all_in
        ]

        if not need_act_indices:
            state.round_over = True
            return

        if all(state.players[i].to_call == 0 for i in need_act_indices) and \
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
def play_one_hand_multi(players: List[Player], button_index: int, hand_no: int = 1, save_dir: str = "logs", game_mode: str = "random") -> int:
    """进行一手牌。返回新的按钮 index (下一手使用)。未实现 side pot。"""
    def _persist():
        try:
            os.makedirs(save_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            path = os.path.join(save_dir, f"hand_{hand_no}_{ts}.json")
            hand_log.save_json(path)
        except Exception as e:
            print(f"保存手牌日志失败: {e}")
    def _summaries():
        # 为启用的玩家生成/更新总结
        for pl in players:
            if getattr(pl, "enable_summary", False):
                generate_player_summary(hand_log, pl)
    # 重置
    for p in players:
        p.reset_for_new_hand()

    if game_mode == "fix":
        # 强制玩家顺序 Max, Lucy
        if len(players) != 2 or players[0].name != "Max" or players[1].name != "Lucy":
            print("[FixMode] 强制重排玩家为 Max, Lucy")
            # 重新构建或报错，这里简单重排（假设名字已经是 Max Lucy）
            players.sort(key=lambda x: ["Max","Lucy"].index(x.name))
        deck = build_fixed_deck()
    else:
        deck = Deck()
        deck.shuffle()

    # 构建 hand_log
    dummy_btn = players[button_index]
    dummy_bb  = players[(button_index+1)%len(players)] if len(players)>1 else players[0]
    hand_log = HandLog(hand_no, button_name=dummy_btn.name, sb=SMALL_BLIND, bb=BIG_BLIND,
                       p_btn=dummy_btn, p_bb=dummy_bb, players=players)
    hand_log.starting_stacks = {p.name: p.stack for p in players}
    hand_log.game_mode = game_mode  # 方便序列化

    # 发底牌
    for p in players:
        p.hole = deck.deal(2)
        hand_log.set_hole_cards(p.name, p.hole)

    state = BettingState(players, button_index, hand_log=hand_log, game_mode=game_mode)
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
    # 若已有人赢下整手（fold 清光）则直接结束（以前这里没有保存日志，导致早期结束的牌未落盘）
    if state.pot == 0 or sum(0 if p.has_folded else 1 for p in players) <=1:
        _summaries()
        _persist()
        return (button_index + 1) % len(players)

    # 翻牌
    deck.burn()
    flop_cards = deck.deal(3)
    board = flop_cards[:]  # 新建一个列表
    hand_log.set_board("Flop", flop_cards)
    print(f"\n翻牌 FLOP: {flop_cards}")
    first_flop = (button_index + 1) % len(players) if len(players)>2 else (state.button_index ^ 1)
    run_betting_round(state, "Flop", first_flop)
    if state.pot == 0 or sum(0 if p.has_folded else 1 for p in players) <=1:
        _summaries()
        _persist()
        return (button_index + 1) % len(players)

    # 转牌
    deck.burn()
    turn_card = deck.deal(1)
    board += turn_card
    hand_log.set_board("Turn", turn_card)
    print(f"\n转牌 TURN: {turn_card}")
    run_betting_round(state, "Turn", first_flop)
    if state.pot == 0 or sum(0 if p.has_folded else 1 for p in players) <=1:
        _summaries()
        _persist()
        return (button_index + 1) % len(players)

    # 河牌
    deck.burn()
    river_card = deck.deal(1)
    board += river_card
    hand_log.set_board("River", river_card)
    print(f"\n河牌 RIVER: {river_card}")
    run_betting_round(state, "River", first_flop)
    if state.pot == 0 or sum(0 if p.has_folded else 1 for p in players) <=1:
        _summaries()
        _persist()
        return (button_index + 1) % len(players)

    # 摊牌
    remaining = [p for p in players if not p.has_folded]
    showdown_multi(remaining, board, state.pot)
    state.pot = 0

    # 摊牌和结算后
    # 记录结果信息
    stacks_before = getattr(hand_log, "stacks_before", None)
    if stacks_before is None:
        stacks_before = {p.name: p.stack for p in players}
    stacks_after = {p.name: p.stack for p in players}
    delta = {name: stacks_after[name] - stacks_before.get(name, 0) for name in stacks_after}
    winners = [name for name, d in delta.items() if d > 0]
    losers = [name for name, d in delta.items() if d < 0]
    result_info = {
        "winners": winners,
        "losers": losers,
        "delta": delta,
        "ending_stacks": stacks_after,
    }
    # 牌型信息（只记录未弃牌玩家）
    hand_types = {}
    # board = ... # 获取当前公共牌列表
    for p in players:
        if not p.has_folded:
            cat, tie, best5 = best_7_of_5(p.hole + board)
            hand_types[p.name] = {
                "category": category_name(cat),
                "best_hand": [str(card) for card in best5],
                "tie_breaker": tie
            }
    result_info["hand_types"] = hand_types
    result_info["board"] = [str(card) for card in board]
    hand_log.result_summary = result_info

    # 保存日志
    _summaries()
    _persist()
    return (button_index + 1) % len(players)

# =========================
# 手牌总结生成（英文输出）
# =========================
SUMMARY_MAX_CHARS = 700

def build_summary_prompt(old_summary: str, hand_log: HandLog, player: Player) -> str:
    # 1) 汇总所有玩家动作（按街道顺序）
    actions_all = []
    for street in ("Preflop","Flop","Turn","River"):
        for a in hand_log.history.get(street, []):
            amt = "" if getattr(a, "amount", 0) in (None, 0) else f" {a.amount}"
            actions_all.append(f"{street}:{a.actor} {a.action}{amt} (pot={a.pot_after})")

    # 2) 自己的手牌与在当前公牌下的牌型（不泄露对手手牌）
    # 获取最终已揭示的公牌
    board_cards = []
    try:
        if hasattr(hand_log, "board") and isinstance(hand_log.board, dict):
            for st in ("Flop","Turn","River"):
                cs = hand_log.board.get(st)
                if cs:
                    # 兼容列表或单张
                    if isinstance(cs, list):
                        board_cards.extend(cs)
                    else:
                        board_cards.append(cs)
        elif hasattr(hand_log, "result_summary") and hand_log.result_summary.get("board"):
            # 兼容 result_summary 中的字符串牌面
            from_str = []
            for s in hand_log.result_summary["board"]:
                # 简单解析形如 'Q♦'
                rank_str, suit = s[:-1], s[-1]
                # 将 10 可能写作 '10'
                rank_map = {"T":10,"J":11,"Q":12,"K":13,"A":14}
                r = rank_map.get(rank_str.upper(), int(rank_str) if rank_str.isdigit() else 0)
                board_cards.append(Card(r, suit))
    except Exception:
        pass

    hero_hole_str = " ".join(str(c) for c in (player.hole or [])) or "(unknown)"
    hero_hand_line = "Hero Hand: n/a"
    try:
        if player.hole and len(player.hole) + len(board_cards) >= 5:
            cat, tie, best5 = best_7_of_5(player.hole + board_cards)
            hero_best_str = " ".join(str(c) for c in best5)
            hero_hand_line = f"Hero Hand: {category_name(cat)} [{hero_best_str}]"
    except Exception:
        pass

    board_str = " ".join(str(c) for c in board_cards) if board_cards else "(none)"

    # 3) 结果与筹码变化
    result = "unknown"
    start_stack = hand_log.starting_stacks.get(player.name, player.stack)
    end_stack = player.stack
    if end_stack > start_stack:
        result = "won chips"
    elif end_stack < start_stack:
        result = "lost chips"
    else:
        result = "no net change"

    # 4) 旧总结
    old_block = old_summary.strip() if old_summary else "(none)"

    # 5) 生成总结用 prompt（英文、精简）
    actions_line = " | ".join(actions_all) if actions_all else "no actions logged"
    prompt = (
        "You are updating a concise personal poker strategy note after a single hand.\n"
        f"Keep it under {SUMMARY_MAX_CHARS} characters.\n"
        "You may ADD new insights, MODIFY existing ones, or REMOVE outdated ones.\n"
        "If no meaningful adjustment is needed, return the previous summary unchanged.\n"
        "Write in English, bullet-like short sentences (no numbering). Avoid repeating obvious rules.\n"
        "Never include lines starting with 'decision:'.\n"
        "--- Previous Summary ---\n" + old_block + "\n"
        "--- This Hand Key Facts ---\n"
        f"Result: {result}; Stack change: {end_stack-start_stack}\n"
        f"Hero hole: {hero_hole_str}\n"
        f"Board: {board_str}\n"
        f"{hero_hand_line}\n"
        f"Action History (all players): {actions_line}\n"
        "--- Output Updated Summary Below ---"
    )
    return prompt

def generate_player_summary(hand_log: HandLog, player: Player):
    old = player.strategy_summary or ""
    prompt = build_summary_prompt(old, hand_log, player)
    try:
        resp, _, _ = player.agent(model_name='gpt-4o-mini', prompt=prompt)
        text = (resp or "").strip()
        # 清洗：去掉围栏与 decision 行、裁剪
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        lines = [ln for ln in text.splitlines() if not ln.lower().startswith("decision:")]
        cleaned = " ".join(l.strip() for l in lines if l.strip())
        if len(cleaned) > SUMMARY_MAX_CHARS:
            cleaned = cleaned[:SUMMARY_MAX_CHARS].rstrip()
        if cleaned and cleaned != old:
            player.strategy_summary = cleaned
            player.strategy_versions.append({"hand_no": str(hand_log.hand_no), "text": cleaned})
            hand_log.strategy_summaries[player.name] = cleaned
            hand_log.summary_updates.append(player.name)
        else:
            # 未变化也记录当前（便于下游）
            hand_log.strategy_summaries[player.name] = old
    except Exception as e:
        hand_log.strategy_summaries[player.name] = old
        hand_log.summary_updates.append(f"error:{player.name}")

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    ################ args ################
    parser.add_argument('--api_batch_use', type=bool, default=False, help='wheather use batch api.')

    args, unparsed = parser.parse_known_args()
    return args
def resume_from_file(path: str, api_args: dict):
    with open(path, "r", encoding="utf-8") as f:
        hand = json.load(f)
    names = hand.get("player_order") or list(hand["starting_stacks"].keys())
    sb = hand.get("sb", hand.get("sb_size", 50))
    bb = hand.get("bb", hand.get("bb_size", 100))
    ending = hand.get("ending_stacks") or hand.get("starting_stacks", {})
    start = hand.get("starting_stacks", {})
    stacks = {n: ending.get(n, start.get(n, 0)) for n in names}

    players = []
    summaries = hand.get("strategy_summaries", {})
    for n in names:
        p = Player(name=n, stack=stacks.get(n, 0), args=api_args)  # 传入 api_args
        if hasattr(p, "strategy_summary"):
            p.strategy_summary = summaries.get(n, "")
        players.append(p)

    last_btn_name = hand.get("button") or hand.get("button_name")
    try:
        last_btn_idx = names.index(last_btn_name)
    except Exception:
        last_btn_idx = 0
    next_button_index = (last_btn_idx + 1) % len(names)
    return players, next_button_index, sb, bb

def resume_from_latest_log(save_dir: str = "logs", api_args: dict = None):
    os.makedirs(save_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(save_dir, "hand_*.json")), key=os.path.getmtime)
    if not files:
        return None
    latest = files[-1]
    print(f"[Resume] Loading latest hand: {os.path.basename(latest)}")
    return resume_from_file(latest, api_args)

def _make_card(code: str) -> Card:
    code = code.strip()
    if len(code) < 2:
        raise ValueError(f"Bad card code: {code}")
    r_str, suit = code[:-1], code[-1]
    r_up = r_str.upper()
    # 处理 T
    if r_up == "T":
        rank = 10
    else:
        if r_up in STR_TO_RANK:
            rank = STR_TO_RANK[r_up]
        else:
            rank = int(r_str)
    suit_map = {'c': '♣', 'd': '♦', 'h': '♥', 's': '♠'}
    suit_sym = suit_map.get(suit.lower(), suit)
    return Card(rank, suit_sym)

def build_fixed_deck() -> Deck:
    """
    固定顺序：
      Button 两张 -> BigBlind 两张 -> burn -> Flop3 -> burn -> Turn1 -> burn -> River1 -> 其余随机
    """
    preset = FIX_MODE_PRESET
    btn_hole = [_make_card(c) for c in preset["button"]["hole"]]
    bb_hole  = [_make_card(c) for c in preset["bigblind"]["hole"]]
    flop = [_make_card(c) for c in preset["board"][:3]]
    turn = [_make_card(preset["board"][3])]
    river = [_make_card(preset["board"][4])]

    # 构建整副牌并排除已用
    full = [Card(r, s) for r in RANKS for s in SUITS]
    used_sig = {repr(c) for c in (btn_hole + bb_hole + flop + turn + river)}
    rest = [c for c in full if repr(c) not in used_sig]
    random.shuffle(rest)

    burn1 = rest.pop()
    burn2 = rest.pop()
    burn3 = rest.pop()

    order = []
    order += btn_hole
    order += bb_hole
    order.append(burn1)
    order += flop
    order.append(burn2)
    order += turn
    order.append(burn3)
    order += river
    order += rest

    dk = Deck()
    dk.cards = order
    return dk
