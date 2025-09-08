from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
import time, json, hashlib
STREETS = ["Preflop", "Flop", "Turn", "River"]
class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP    = "flop"
    TURN    = "turn"
    RIVER   = "river"
@dataclass
class LegalAction:
    type: str                      # 'fold'|'check'|'call'|'raise'|'all-in'  (统一，不再使用 bet)
    to: Optional[int] = None       # call 的目标数额
    min_to: Optional[int] = None   # bet/raise 的下界（to）
    max_to: Optional[int] = None   # bet/raise 的上界（to）
@dataclass
class Observation:  # 直接存你喂给 LLM 的“状态快照”（便于复现）
    players: List[str]
    hero: str
    hole_cards: List[str]
    street: str
    button: str
    blinds: Dict[str, int]
    board: List[str]               # 统一一个 board 字段（preflop 为空）
    pot: int
    to_act: str
    to_call: int
    positions: Dict[str, str]      # 可选：BTN/SB/BB/UTG...
    history_text: List[str]        # 你现在生成的 History 行（可复现 prompt）
    legal_actions: List[LegalAction]
    strategy_notes: Optional[str] = ""   # 该玩家当前策略总结
    all_strategy_summaries: Optional[Dict[str, str]] = None  # 所有玩家最新总结（仅内部使用）
    strategy_history: Optional[List[str]] = None  # 该玩家历史版本文本（旧->新 或 截断）

    def to_prompt_text(self) -> str:
        # 你现有 build_llm_prompt_text 的更“稳”版本（裁剪为必需字段）
        lines = []
        lines.append(f"Players: {self.players} | Hero: {self.hero}")
        lines.append(f"Hole cards: {self.hole_cards} | Street: {self.street}")
        lines.append(f"Button: {self.button} | Blinds: {self.blinds['sb']}/{self.blinds['bb']}")
        lines.append(f"Board: {self.board} | Pot: {self.pot} | To act: {self.to_act} | To call: {self.to_call}")
        if self.positions:
            lines.append(f"Positions: {self.positions}")
        if self.strategy_notes:
            lines.append("Player Strategy Notes (current):")
            lines.append(self.strategy_notes.strip())
            lines.append("---")
        if self.strategy_history:
            lines.append("Player Strategy History (older to newer):")
            for h in self.strategy_history:
                lines.append(f"  - {h}")
            lines.append("---")
    # 全部玩家策略总结暂时关闭（保留数据但不展示）
    # if self.all_strategy_summaries:
    #     lines.append("All Players Strategy Summaries (private internal context):")
    #     for n, s in self.all_strategy_summaries.items():
    #         if not s:
    #             continue
    #         lines.append(f"  {n}: {s[:300]}")
    #     lines.append("---")
        lines.append("History:")
        for h in self.history_text: lines.append(f"  {h}")
        # 动作域（强约束提示）
        def _fmt(a: LegalAction):
            if a.type == "raise":  # 首次进攻也用 raise 表达
                lo = f"min:{a.min_to}" if a.min_to is not None else ""
                hi = f"max:{a.max_to}" if a.max_to is not None else ""
                rng = f" [{lo} {hi}]".strip()
                return f"{a.type}{rng}"
            if a.type == "call" and a.to is not None:
                return f"call to {a.to}"
            return a.type
        acts = " | ".join(_fmt(a) for a in self.legal_actions)
        lines.append(f"Legal actions: {acts}")
        # 输出规范
        lines.append(
            "Reason explicitly in natural language.\n"
            "Then output exactly ONE final line, in lowercase, starting at column 1, as:\n"
            "decision: <action>\n\n"
            "Rules:\n"
            "- Do NOT include any other line that starts with \"decision:\" before the final line.\n"
            "- <action> ∈ fold | check | call | raise [total] | all-in\n"
            "- Treat ANY first aggression as 'raise' <total> (do not use 'bet')."
            "- '<total>' indicates your total contribution in this round (including previous chips)."
            "If you had not invested any chips before, 'raise 300' means your first bet of 300."
            "- Use integer chip units (no symbols, no commas).\n"
            "Important strategy note:\n"
            "Do not always choose 'check'.\n"
            "You should estimate your winning probability (equity) based on your hole cards and the board.\n"
            "If your winning chance is high, increase aggression via raises.\n"
            "If your winning chance is low, fold more often.\n"
            "You can also consider bluffing: when the opponent's betting pattern suggests they may be weak, you may raise to apply pressure.\n"
        )
        return "\n".join(lines)
@dataclass
class ParsedDecision:
    action: str                    # 'fold'|'check'|'call'|'bet'|'raise'|'all-in'
    amount: Optional[int] = None   # bet 的金额或 all-in for N（可空）
    to: Optional[int] = None       # raise 的 to 总额
    text: str = ""                 # 'decision:' 后半原文（如 'raise to 200'）
        
@dataclass
class ActionEntry:
    street: str               # Preflop/Flop/Turn/River
    actor: str                # 玩家名
    action: str               # fold/check/call/bet/raise/all-in/SB/BB
    amount: int               # 本次动作“新增投进去”的筹码（对 fold/check = 0）
    to_call_before: int       # 动作前该玩家的 to_call
    invested_before: int      # 动作前该玩家在本轮已投入
    invested_after: int       # 动作后该玩家在本轮已投入
    pot_after: int            # 动作后底池
    note: str = ""            # 可选备注（例如“翻前补到BB”）

@dataclass
class HandLog:
    """记录一手牌的所有信息（已扩展支持多玩家 steps 日志）。"""
    hand_no: int
    button_name: str
    sb_size: int
    bb_size: int
    starting_stacks: Dict[str, int]
    hole_cards: Dict[str, List[str]]
    board: Dict[str, List[str]]
    history: Dict[str, List[ActionEntry]]
    strategy_summaries: Dict[str, str] = field(default_factory=dict)  # 每手结束后的各玩家总结（启用者）
    summary_updates: List[str] = field(default_factory=list)          # 本手实际发生变化的玩家名列表
    steps: List[Any] = field(default_factory=list)  # 保存 StepTrace（避免循环依赖，使用 Any）

    def __init__(self, hand_no: int, button_name: str, sb: int, bb: int, p_btn, p_bb, players: Optional[List[Any]] = None):
        self.hand_no = hand_no
        self.button_name = button_name
        self.sb_size = sb
        self.bb_size = bb
        # 若传入 players 列表则全部记录初始筹码；否则保留双人兼容
        if players:
            self.starting_stacks = {p.name: p.stack for p in players}
        else:
            self.starting_stacks = {p_btn.name: p_btn.stack, p_bb.name: p_bb.stack}
        self.hole_cards = {}
        self.board = {s: [] for s in STREETS if s != "Preflop"}
        self.history = {s: [] for s in STREETS}
        self.steps = []
        self.strategy_summaries = {}
        self.summary_updates = []

    def set_hole_cards(self, p_name: str, cards) -> None:
        self.hole_cards[p_name] = [repr(c) for c in cards]

    def set_board(self, street: str, cards_list) -> None:
        self.board[street] = [repr(c) for c in cards_list]

    def add_action(self, entry: ActionEntry) -> None:
        self.history[entry.street].append(entry)

    def add_step(self, step) -> None:
        self.steps.append(step)

    def export_compact_state(self, current_street: str) -> Dict:
        return {
            "hand_no": self.hand_no,
            "street": current_street,
            "button": self.button_name,
            "sb": self.sb_size,
            "bb": self.bb_size,
            "starting_stacks": self.starting_stacks,
            "hole_cards": self.hole_cards,
            "board": self.board,
            "history": {s: [asdict(h) for h in self.history[s]] for s in STREETS},
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hand_no": self.hand_no,
            "button": self.button_name,
            "sb": self.sb_size,
            "bb": self.bb_size,
            "starting_stacks": self.starting_stacks,
            "hole_cards": self.hole_cards,
            "board": self.board,
            "history": {s: [asdict(h) for h in self.history[s]] for s in STREETS},
            "strategy_summaries": self.strategy_summaries,
            "summary_updates": self.summary_updates,
            "steps": [asdict(s) for s in self.steps],
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def save_jsonl(self, path: str):
        with open(path, "a", encoding="utf-8") as f:
            f.write(self.to_json().replace("\n", " ") + "\n")
@dataclass  # duplicate kept earlier; ensure single definition only (上面已有 ParsedDecision 定义, 此处保持 StepTrace)
class StepTrace:
    idx: int
    street: str
    actor: str
    observation: Observation
    prompt_hash: str
    prompt_text: Optional[str]
    model: str
    response_text: str
    decision: ParsedDecision
    # 应用决策前后状态快照（便于对账/回放）
    pot_before: int
    pot_after: int
    stacks_before: Dict[str, int]
    stacks_after: Dict[str, int]
    invested_before: Dict[str, int]
    invested_after: Dict[str, int]
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    analysis_saved: Optional[str] = None
    analysis_chars: int = 0
    analysis_sha256: Optional[str] = None  
## 旧的游离函数已内联到 HandLog；保留占位避免外部误引用
def build_llm_prompt_text(*args, **kwargs):
    raise NotImplementedError("build_llm_prompt_text 已弃用，使用 Observation.to_prompt_text()")

