# poker/recorder.py
from __future__ import annotations
from typing import Callable, Optional, Dict, List
import hashlib, json, unicodedata, re
from dataclasses import asdict
from models import Observation, ParsedDecision, StepTrace, HandLog, LegalAction

_DECISION_LINE_RE = re.compile(r"(?im)^\s*decision\s*[:：]\s*(?P<raw>.+?)\s*$")
def split_analysis_and_decision(response_text: str):
    """
    返回 (analysis_text, decision_line_text, decision_body_text)
    - 若找不到 decision 行，analysis_text=全文，其余为 None
    """
    t = unicodedata.normalize("NFKC", response_text or "")
    ms = list(_DECISION_LINE_RE.finditer(t))
    if not ms:
        return t, None, None
    m = ms[-1]
    analysis = t[:m.start()].rstrip("\n")
    return analysis, m.group(0).strip(), m.group("raw").strip()   
# 1) 根据 BettingState + actor 生成 Observation（你已有的信息填进去）
def make_observation(state, actor_name: str) -> Observation:
    hl = getattr(state, "hand_log", None)

    # --- players 列表 ---
    if hl and getattr(hl, "starting_stacks", None):
        players = list(hl.starting_stacks.keys())
    elif getattr(state, "players", None):
        players = [getattr(p, "name", "") for p in state.players if getattr(p, "name", None)]
    else:
        players = []

    # --- blinds ---
    blinds: Dict[str, int] = {
        "sb": int(getattr(state, "sb_size", 0) or 0),
        "bb": int(getattr(state, "bb_size", 0) or 0),
        "ante": int(getattr(state, "ante", 0) or 0),
    }
    # 若 state 无，尝试从 hand_log.game.blinds 回退
    if (blinds["sb"] == 0 and blinds["bb"] == 0) and hl and getattr(hl, "game", None):
        gb = hl.game.get("blinds", {}) if isinstance(hl.game, dict) else {}
        blinds["sb"] = int(gb.get("sb", 0))
        blinds["bb"] = int(gb.get("bb", 0))
        blinds["ante"] = int(gb.get("ante", 0))

    # --- board ---
    board: List[str] = []
    bd = getattr(state, "board", None)
    if isinstance(bd, dict):
        board += bd.get("Flop", []) or []
        board += bd.get("Turn", []) or []
        board += bd.get("River", []) or []
    elif hl and getattr(hl, "board_by_street", None):
        bbs = hl.board_by_street
        # 兼容大小写键
        board += (bbs.get("flop") or bbs.get("Flop") or []) \
              +  (bbs.get("turn") or bbs.get("Turn") or []) \
              +  (bbs.get("river") or bbs.get("River") or [])

    # --- history 文本（用于复现 prompt）---
    history_lines: List[str] = []
    hist = getattr(state, "history", None)
    if isinstance(hist, dict):
        for s in ("Preflop", "Flop", "Turn", "River"):
            for h in hist.get(s, []) or []:
                actor  = getattr(h, "actor", "?")
                action = getattr(h, "action", "?")
                amount = getattr(h, "amount", None)
                pot_a  = getattr(h, "pot_after", None)
                amt_s  = "" if amount in (None, 0) else f" {amount}"
                pot_s  = f" (pot={pot_a})" if pot_a is not None else ""
                history_lines.append(f"[{s}] {actor} {action}{amt_s}{pot_s}")
    elif hl and getattr(hl, "steps", None):
        for st in hl.steps:
            history_lines.append(f"[{str(st.street).upper()}] {st.actor} {getattr(st.decision, 'text', '')}")

    # --- positions（可选，没有就空）---
    positions = getattr(state, "positions", {}) or {}

    # --- hero 手牌 ---
    hero_cards: List[str] = []
    hc = getattr(state, "hole_cards", None)
    if isinstance(hc, dict) and actor_name in hc:
        hero_cards = hc[actor_name]
    elif hl and getattr(hl, "hole_cards", None) and actor_name in hl.hole_cards:
        hero_cards = hl.hole_cards[actor_name]

    # --- 其它基本量 ---
    street = str(getattr(state, "current_street", "preflop")).lower()
    button = getattr(state, "button_name", None) or (getattr(hl, "button", "") if hl else "")
    pot    = int(getattr(state, "pot", 0) or 0)

    # to_call
    to_call = 0
    pmap = getattr(state, "players_by_name", None)
    if isinstance(pmap, dict) and actor_name in pmap:
        to_call = int(getattr(pmap[actor_name], "to_call", 0) or 0)
    else:
        for p in getattr(state, "players", []) or []:
            if getattr(p, "name", None) == actor_name:
                to_call = int(getattr(p, "to_call", 0) or 0)
                break

    # --- legal_actions（首选引擎 API；无则最小回退）---
    if hasattr(state, "get_legal_actions_for"):
        legal_actions: List[LegalAction] = list(state.get_legal_actions_for(actor_name))
    else:
        # 仅用于提示，不做严格边界；你的执行校验还是走规则引擎
        legal_actions = []
        if to_call > 0:
            legal_actions.extend([LegalAction("fold"), LegalAction("call"), LegalAction("raise")])
        else:
            legal_actions.extend([LegalAction("check"), LegalAction("bet")])

    return Observation(
        players=players,
        hero=actor_name,
        hole_cards=hero_cards or [],
        street=street,
        button=button or "",
        blinds=blinds,
        board=board,
        pot=pot,
        to_act=actor_name,
        to_call=to_call,
        positions=positions,
        history_text=history_lines,
        legal_actions=legal_actions,
    )
def _snapshot(state):
    # 名字列表（用于把 invested 列表映射成字典）
    players = getattr(state, "players", [])
    names = [getattr(p, "name", str(i)) for i, p in enumerate(players)]

    # 处理 stacks
    stacks = {names[i]: getattr(players[i], "stack", 0) for i in range(len(players))}

    # 处理 invested：兼容 dict / list / tuple
    raw_inv = getattr(state, "invested", None)
    if isinstance(raw_inv, dict):
        invested = dict(raw_inv)
    elif isinstance(raw_inv, (list, tuple)):
        invested = {names[i]: raw_inv[i] for i in range(min(len(names), len(raw_inv)))}
    else:
        invested = {}

    return {
        "pot": int(getattr(state, "pot", 0) or 0),
        "stacks": stacks,
        "invested": invested,
    }

class StepRecorder:
    """
    负责：抓 before/after 快照、生成 StepTrace、写入 HandLog（和可选JSONL）
    不负责：规则判断、LLM 调用本身。
    """
    def __init__(self,
                 hand_log: HandLog,
                 state,
                 actor_name: str,
                 model_name: str,
                 obs: Observation,
                 response_text: str,
                 decision: ParsedDecision,
                 jsonl_path: Optional[str] = None,
                 store_response: bool = True,
                 store_analysis: bool = True,
                 analysis_max_chars: int = 4000):
        self.hand_log = hand_log
        self.state = state
        self.actor = actor_name
        self.model = model_name
        self.obs = obs
        self.response_text = response_text if store_response else ""
        self.decision = decision
        self.jsonl_path = jsonl_path
        self.store_analysis = store_analysis
        self.analysis_max_chars = analysis_max_chars

    def run(self, apply_fn: Callable[[], None]) -> StepTrace:
        # 1) before 快照
        before = _snapshot(self.state)

        # 2) 应用动作（由调用方提供，避免 recorder 依赖引擎）
        apply_fn()

        # 3) after 快照
        after = _snapshot(self.state)

        # 4) 组 StepTrace
        prompt_text = self.obs.to_prompt_text()
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]

        analysis_text, decision_line, decision_raw = split_analysis_and_decision(self.response_text)
        clipped = (analysis_text[-4000:] if analysis_text else None)  # 例如仅保留末尾 4000 字
        analysis_sha = hashlib.sha256(analysis_text.encode("utf-8")).hexdigest() if analysis_text else None

        step = StepTrace(
            idx=len(self.hand_log.steps),
            street=self.obs.street,
            actor=self.actor,
            observation=self.obs,
            prompt_hash=prompt_hash,
            model=self.model,
            response_text=self.response_text,    # 已含“思考过程”
            decision=self.decision,
            pot_before=before["pot"], pot_after=after["pot"],
            stacks_before=before["stacks"], stacks_after=after["stacks"],
            invested_before=before["invested"], invested_after=after["invested"],
            analysis_saved=clipped,
            analysis_chars=len(analysis_text) if analysis_text else 0,
            analysis_sha256=analysis_sha,            
        )

        # 5) 入 HandLog
        self.hand_log.add_step(step)

        # 6) 可选：追加写 JSONL（便于训练/回放）
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(step), ensure_ascii=False) + "\n")

        return step
def record_step(
    hand_log: HandLog,
    state, actor_name: str,
    model_name: str,
    response_text: str,
    decision: ParsedDecision,
    obs: Observation
):
    # 仅记录（不在这里执行动作，动作已经由上层引擎处理）
    pot_before      = state.pot
    stacks_before   = {p.name: p.stack for p in state.players}
    invested_before = dict(state.invested)
    pot_after       = pot_before
    stacks_after    = dict(stacks_before)
    invested_after  = dict(invested_before)

    # 生成 step 记录
    prompt_text = obs.to_prompt_text()
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]

    step = StepTrace(
        idx=len(hand_log.steps),
        street=obs.street,
        actor=actor_name,
        observation=obs,
        prompt_hash=prompt_hash,
        model=model_name,
        response_text=response_text,
    decision=decision,
    pot_before=pot_before, pot_after=pot_after,
    stacks_before=stacks_before, stacks_after=stacks_after,
    invested_before=invested_before, invested_after=invested_after,
    )
    if hasattr(hand_log, "add_step"):
        hand_log.add_step(step)