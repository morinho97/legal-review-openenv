"""
legal_env.py — LegalReviewEnv (flat single file, OpenEnv compliant)
ALL scores and metrics strictly in open interval (0,1) — never 0.0 or 1.0.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np


# ── Enums ─────────────────────────────────────────────────────────────────────

class Action(IntEnum):
    APPROVE          = 0
    FLAG_CLAUSE      = 1
    REDLINE          = 2
    REQUEST_CLARIFY  = 3
    ESCALATE_COUNSEL = 4

ACTION_LABELS = {int(a): a.name.lower() for a in Action}
N_ACTIONS     = len(Action)


class RiskLevel(IntEnum):
    NONE     = 0
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


# ── Constants ──────────────────────────────────────────────────────────────────

CLAUSE_TYPES = [
    "indemnity", "limitation_of_liability", "payment_terms",
    "termination", "ip_ownership", "confidentiality",
    "dispute_resolution", "governing_law", "warranty", "data_protection",
]
JURISDICTIONS   = ["US-NY", "US-CA", "UK", "EU-GDPR", "SG", "IN"]
HIGH_RISK_TYPES = {"indemnity", "limitation_of_liability", "termination", "ip_ownership"}
OBS_DIM         = 24

DIFFICULTY_PRESETS: dict[str, dict] = {
    "easy":   dict(n_clauses=10,  contract_type="NDA",               time_budget=None, seed=1001),
    "medium": dict(n_clauses=50,  contract_type="SaaS Agreement",    time_budget=80,   seed=2002),
    "hard":   dict(n_clauses=120, contract_type="M&A Due Diligence", time_budget=150,  seed=3003),
}

SCORE_WEIGHTS   = {"f1": 0.60, "recall": 0.25, "efficiency": 0.15}
MAX_REWARD_STEP = {"easy": 2.0, "medium": 1.95, "hard": 1.95}
TASK_WEIGHTS    = {"easy": 0.20, "medium": 0.35, "hard": 0.45}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Clause:
    clause_id:          int
    text:               str
    clause_type:        str
    jurisdiction:       str
    true_risk:          RiskLevel
    contains_liability: bool
    has_nested_ref:     bool
    prior_redlines:     int = 0


@dataclass
class RewardConfig:
    correct_flag:     float = +2.0
    correct_approve:  float = +1.0
    missed_liability: float = -5.0
    false_positive:   float = -0.2
    redline_cost:     float = -0.3
    escalate_cost:    float = -0.5
    escalate_bonus:   float = +0.8
    time_penalty:     float = -0.05


# ── Templates & scenarios ──────────────────────────────────────────────────────

_TEMPLATES = {
    "indemnity": (
        "Party A shall indemnify, defend and hold harmless Party B from any claims, "
        "damages or liabilities arising out of {scenario}, including reasonable "
        "attorneys fees, to the maximum extent permitted by {jurisdiction} law."
    ),
    "limitation_of_liability": (
        "In no event shall either party be liable for indirect, incidental, or "
        "consequential damages arising out of {scenario}. Total liability shall not exceed {amount}."
    ),
    "payment_terms": (
        "Invoices are due within {days} days of issuance. Late payments accrue "
        "interest at {rate}% per month. Disputes must be raised within {notice_days} days."
    ),
    "termination": (
        "Either party may terminate upon {notice} days written notice. Termination "
        "for cause takes effect if breach is not cured within {cure_days} days."
    ),
    "confidentiality": (
        "Each party shall maintain confidentiality of disclosed information for "
        "{years} years and shall not share with third parties without written consent."
    ),
    "ip_ownership": (
        "All IP developed by Party A in connection with {scenario} shall be the "
        "exclusive property of {owner}. Party B receives a non-exclusive licence only."
    ),
    "dispute_resolution": (
        "All disputes shall be resolved by binding arbitration under the rules of "
        "{body}, seated in {city}. The arbitration language shall be English."
    ),
    "governing_law": (
        "This agreement shall be governed by the laws of {jurisdiction}, without "
        "regard to conflict-of-law provisions."
    ),
    "warranty": (
        "Party A warrants that services will be performed professionally. ALL OTHER "
        "WARRANTIES ARE DISCLAIMED TO THE FULLEST EXTENT PERMITTED BY LAW."
    ),
    "data_protection": (
        "Party A shall process personal data in accordance with {jurisdiction} data "
        "protection law. A Data Processing Agreement is incorporated as Schedule {schedule}."
    ),
}

_SCENARIOS = [
    "synergize scalable markets", "leverage agile frameworks",
    "matrix B2B deliverables", "orchestrate cross-platform content",
    "disintermediate granular paradigms", "productize end-to-end channels",
    "recontextualize real-time supply chains", "integrate extensible mindshare",
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _sc(v: float) -> float:
    """
    Safe-clamp: force any float into the OPEN interval (0.0, 1.0).
    The OpenEnv evaluator rejects exactly 0.0 AND exactly 1.0.
    """
    try:
        v = float(v)
    except Exception:
        return 0.0001
    if v != v:          # NaN
        return 0.0001
    if v <= 0.0:
        return 0.0001
    if v >= 1.0:
        return 0.9999
    r = round(v, 4)
    if r <= 0.0:
        return 0.0001
    if r >= 1.0:
        return 0.9999
    return r

# Public alias used by inference.py and server.py
_clamp = _sc


def _build_clauses(n: int, difficulty: str, seed: int) -> list[Clause]:
    random.seed(seed)
    clauses = []
    for i in range(n):
        ctype = random.choice(CLAUSE_TYPES)
        jur   = random.choice(JURISDICTIONS)
        text  = _TEMPLATES.get(ctype, "Placeholder clause for {scenario}.").format(
            scenario=random.choice(_SCENARIOS), jurisdiction=jur,
            amount=f"${random.randint(10, 500) * 1000:,}",
            days=random.choice([15, 30, 45, 60]),
            rate=round(random.uniform(0.5, 3.0), 1),
            notice_days=random.choice([5, 10, 15, 30]),
            notice=random.choice([30, 60, 90]),
            cure_days=random.choice([10, 15, 30]),
            years=random.choice([2, 3, 5]),
            owner=random.choice(["Party A", "Party B", "the Client"]),
            body=random.choice(["AAA", "ICC", "SIAC", "LCIA"]),
            city=random.choice(["New York", "London", "Singapore", "Paris"]),
            schedule=random.choice(["1", "A", "B"]),
        )
        if ctype in HIGH_RISK_TYPES:
            true_risk, contains_liability = random.choice([RiskLevel.HIGH, RiskLevel.CRITICAL]), True
        elif ctype in {"payment_terms", "data_protection"}:
            true_risk, contains_liability = RiskLevel.MEDIUM, random.random() < 0.3
        else:
            true_risk          = random.choice([RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM])
            contains_liability = False

        has_nested = (difficulty == "hard") and (random.random() < 0.25)
        if has_nested:
            text += f" (See Clause {random.randint(0, max(0, i - 1))} for definitions.)"

        clauses.append(Clause(
            clause_id=i, text=text, clause_type=ctype, jurisdiction=jur,
            true_risk=true_risk, contains_liability=contains_liability,
            has_nested_ref=has_nested,
            prior_redlines=random.randint(0, 3) if difficulty != "easy" else 0,
        ))
    return clauses


def _clause_to_obs(clause: Clause, time_remaining: int, queue_size: int) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = min(time_remaining / 200.0, 1.0)
    obs[1] = min(queue_size     / 200.0, 1.0)
    obs[2] = min(clause.prior_redlines / 10.0, 1.0)
    obs[3] = float(clause.has_nested_ref)
    if clause.clause_type in CLAUSE_TYPES:
        obs[4 + CLAUSE_TYPES.index(clause.clause_type)] = 1.0
    if clause.jurisdiction in JURISDICTIONS:
        obs[14 + JURISDICTIONS.index(clause.jurisdiction)] = 1.0
    return obs


# ── Main environment ───────────────────────────────────────────────────────────

class LegalReviewEnv:
    """
    OpenEnv-compliant legal clause review environment.

    obs                     = env.reset()
    obs, reward, done, info = env.step(action)
    snapshot                = env.state()
    """

    def __init__(self, difficulty: str = "easy", reward_config: Optional[RewardConfig] = None):
        assert difficulty in DIFFICULTY_PRESETS, f"Unknown difficulty: {difficulty}"
        self.difficulty    = difficulty
        self.preset        = DIFFICULTY_PRESETS[difficulty]
        self.reward_config = reward_config or RewardConfig()
        self.n_actions     = N_ACTIONS
        self.obs_dim       = OBS_DIM
        self._clauses:         list[Clause] = []
        self._cursor:          int          = 0
        self._time_remaining:  int          = 0
        self._episode_rewards: list[float]  = []
        self._decisions:       list[dict]   = []
        self._done:            bool         = False

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        eff_seed = seed if seed is not None else self.preset["seed"]
        random.seed(eff_seed)
        np.random.seed(eff_seed)
        self._clauses         = _build_clauses(self.preset["n_clauses"], self.difficulty, eff_seed)
        self._cursor          = 0
        self._time_remaining  = self.preset["time_budget"] or 10_000
        self._episode_rewards = []
        self._decisions       = []
        self._done            = False
        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode finished. Call reset().")
        if not (0 <= action < N_ACTIONS):
            raise ValueError(f"Invalid action {action}. Must be 0-{N_ACTIONS - 1}.")
        clause = self._clauses[self._cursor]
        reward = self._compute_reward(clause, Action(action))
        self._episode_rewards.append(reward)
        self._decisions.append({
            "clause_id":    clause.clause_id,
            "clause_type":  clause.clause_type,
            "jurisdiction": clause.jurisdiction,
            "action":       ACTION_LABELS[action],
            "true_risk":    clause.true_risk.name,
            "reward":       round(reward, 4),
        })
        self._cursor += 1
        if self.preset["time_budget"]:
            self._time_remaining -= 1
        done = (
            self._cursor >= len(self._clauses)
            or (self.preset["time_budget"] is not None and self._time_remaining <= 0)
        )
        self._done = done
        return self._get_obs(), reward, done, self._get_info()

    def state(self) -> dict:
        current = None
        if self._cursor < len(self._clauses):
            c = self._clauses[self._cursor]
            current = {
                "clause_id":          c.clause_id,
                "clause_type":        c.clause_type,
                "jurisdiction":       c.jurisdiction,
                "true_risk":          c.true_risk.name,
                "contains_liability": c.contains_liability,
                "has_nested_ref":     c.has_nested_ref,
                "prior_redlines":     c.prior_redlines,
                "text":               c.text,
            }
        return {
            "difficulty":        self.difficulty,
            "contract_type":     self.preset["contract_type"],
            "total_clauses":     len(self._clauses),
            "reviewed":          self._cursor,
            "time_remaining":    self._time_remaining,
            "done":              self._done,
            "cumulative_reward": round(sum(self._episode_rewards), 4),
            "current_clause":    current,
            "decisions":         list(self._decisions),
        }

    def episode_metrics(self) -> dict:
        """
        Returns per-episode classification metrics.
        f1, precision, recall are ALL clamped to strictly (0, 1).
        """
        if not self._decisions:
            return {
                "f1": 0.0001, "precision": 0.0001, "recall": 0.0001,
                "true_positives": 0, "false_positives": 0, "false_negatives": 0,
                "total_reward": 0.0, "missed_liability_clauses": 0,
            }
        flag_actions = {"flag_clause", "redline", "escalate_counsel"}
        flagged = {d["clause_id"] for d in self._decisions if d["action"] in flag_actions}
        risky   = {c.clause_id for c in self._clauses
                   if c.true_risk >= RiskLevel.HIGH or c.contains_liability}
        tp = len(flagged & risky)
        fp = len(flagged - risky)
        fn = len(risky - flagged)
        raw_precision = tp / (tp + fp) if (tp + fp) else 0.0
        raw_recall    = tp / (tp + fn) if (tp + fn) else 0.0
        raw_f1        = (2 * raw_precision * raw_recall / (raw_precision + raw_recall)
                         if (raw_precision + raw_recall) else 0.0)
        # *** Clamp all ratio metrics — evaluator rejects 0.0 and 1.0 ***
        return {
            "f1":                      _sc(raw_f1),
            "precision":               _sc(raw_precision),
            "recall":                  _sc(raw_recall),
            "true_positives":          tp,
            "false_positives":         fp,
            "false_negatives":         fn,
            "total_reward":            round(sum(self._episode_rewards), 4),
            "missed_liability_clauses": fn,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_reward(self, clause: Clause, action: Action) -> float:
        rc, r = self.reward_config, 0.0
        if self.preset["time_budget"]:
            r += rc.time_penalty
        if action == Action.APPROVE:
            r += (rc.missed_liability
                  if (clause.contains_liability or clause.true_risk >= RiskLevel.HIGH)
                  else rc.correct_approve)
        elif action == Action.FLAG_CLAUSE:
            r += (rc.correct_flag
                  if (clause.true_risk >= RiskLevel.HIGH or clause.contains_liability)
                  else rc.false_positive)
        elif action == Action.REDLINE:
            r += ((rc.correct_flag * 0.7 if clause.true_risk >= RiskLevel.MEDIUM
                   else rc.false_positive) + rc.redline_cost)
        elif action == Action.REQUEST_CLARIFY:
            r += (rc.correct_approve * 0.5
                  if (clause.has_nested_ref or clause.true_risk == RiskLevel.MEDIUM)
                  else rc.false_positive * 0.5)
        elif action == Action.ESCALATE_COUNSEL:
            r += rc.escalate_bonus + rc.escalate_cost
            if clause.true_risk >= RiskLevel.CRITICAL:
                r += rc.correct_flag
        return r

    def _get_obs(self) -> np.ndarray:
        if self._cursor >= len(self._clauses):
            return np.zeros(OBS_DIM, dtype=np.float32)
        return _clause_to_obs(
            self._clauses[self._cursor],
            self._time_remaining,
            len(self._clauses) - self._cursor,
        )

    def _get_info(self) -> dict:
        fa = {"flag_clause", "redline", "escalate_counsel"}
        return {
            "reviewed":       self._cursor,
            "remaining":      len(self._clauses) - self._cursor,
            "n_flagged":      sum(1 for d in self._decisions if d["action"] in fa),
            "n_approved":     sum(1 for d in self._decisions if d["action"] == "approve"),
            "time_remaining": self._time_remaining,
            "total_reward":   round(sum(self._episode_rewards), 4),
        }


# ── Grader (module-level) ──────────────────────────────────────────────────────

def grade_episode(env: LegalReviewEnv) -> float:
    """Score a completed episode. Returns strictly (0, 1) — never 0.0 or 1.0."""
    m = env.episode_metrics()
    if env._cursor == 0:
        return 0.0001
    max_possible = MAX_REWARD_STEP[env.difficulty] * env._cursor
    efficiency   = max(0.0, min(1.0, m["total_reward"] / max_possible)) if max_possible else 0.0
    raw = (SCORE_WEIGHTS["f1"]         * m["f1"] +
           SCORE_WEIGHTS["recall"]     * m["recall"] +
           SCORE_WEIGHTS["efficiency"] * efficiency)
    return _sc(raw)


def grade_all_tasks(results: dict) -> dict:
    """Aggregate per-task results. Every score strictly (0, 1)."""
    per_task    = {t: _sc(results[t]["score"]) for t in results}
    raw_final   = sum(TASK_WEIGHTS.get(t, 1 / 3) * per_task[t] for t in per_task)
    final_score = _sc(raw_final)
    return {
        "per_task":    per_task,
        "final_score": final_score,
        "passed":      raw_final >= 0.40,
    }


# ── Rule-based agent ───────────────────────────────────────────────────────────

_HIGH_RISK = {"indemnity", "limitation_of_liability", "termination", "ip_ownership"}


def rule_based_action(ctx: dict) -> int:
    if ctx.get("time_remaining", 999) < 10:
        return int(Action.ESCALATE_COUNSEL)
    if ctx.get("clause_type", "") in _HIGH_RISK:
        return int(Action.FLAG_CLAUSE)
    if ctx.get("has_nested_ref", False):
        return int(Action.REQUEST_CLARIFY)
    if (ctx.get("clause_type", "") in {"payment_terms", "data_protection"}
            and ctx.get("prior_redlines", 0) > 1):
        return int(Action.REDLINE)
    return int(Action.APPROVE)