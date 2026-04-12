"""
inference.py — OpenEnv Hackathon inference script
Strict log format: [START] / [STEP] / [END]
ALL scores and metrics strictly in open interval (0, 1) — never 0.0 or 1.0.
"""
from __future__ import annotations
import json, os
from openai import OpenAI
from legal_env import (
    LegalReviewEnv, N_ACTIONS, ACTION_LABELS,
    grade_episode, grade_all_tasks, rule_based_action, _clamp,
)

API_BASE_URL     = os.getenv("API_BASE_URL",  "http://localhost:8000/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",    "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "EMPTY")

_SYS = ("You are a senior contract lawyer. Given clause info, pick ONE action:\n"
        "0=APPROVE 1=FLAG_CLAUSE 2=REDLINE 3=REQUEST_CLARIFY 4=ESCALATE_COUNSEL\n"
        "Reply with ONLY a single digit 0-4.")


def _llm_action(ctx: dict) -> int:
    prompt = (f"Contract:{ctx.get('contract_type','')} Type:{ctx.get('clause_type','')}\n"
              f"Jur:{ctx.get('jurisdiction','')} Nested:{ctx.get('has_nested_ref',False)}\n"
              f"Redlines:{ctx.get('prior_redlines',0)} Time:{ctx.get('time_remaining',999)}\nAction:")
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user",   "content": prompt}],
            max_tokens=3, temperature=0.0)
        a = int(resp.choices[0].message.content.strip()[0])
        return a if 0 <= a < N_ACTIONS else rule_based_action(ctx)
    except Exception:
        return rule_based_action(ctx)


def _safe_metrics(m: dict) -> dict:
    """Clamp f1/precision/recall and rebuild metrics dict cleanly."""
    return {
        "f1":                      _clamp(m["f1"]),
        "precision":               _clamp(m["precision"]),
        "recall":                  _clamp(m["recall"]),
        "true_positives":          m["true_positives"],
        "false_positives":         m["false_positives"],
        "false_negatives":         m["false_negatives"],
        "total_reward":            m["total_reward"],
        "missed_liability_clauses": m["missed_liability_clauses"],
    }


def run_task(task_name: str, use_llm: bool = True) -> dict:
    env  = LegalReviewEnv(difficulty=task_name)
    obs  = env.reset()
    print(f"[START] Task: {task_name}")
    step_num, done = 0, False
    while not done:
        s   = env.state(); ctx = s.get("current_clause") or {}
        ctx["contract_type"]  = s.get("contract_type", "")
        ctx["time_remaining"] = s.get("time_remaining", 999)
        action = _llm_action(ctx) if use_llm else rule_based_action(ctx)
        obs, reward, done, info = env.step(action)
        step_num += 1
        print(f"[STEP]  step={step_num}, reward={reward:.4f}")
    score = grade_episode(env)
    # Extra safety clamp (should already be clamped, but defensive)
    if score <= 0.0:
        score = 0.0001
    elif score >= 1.0:
        score = 0.9999
    else:
        score = round(score, 4)
        if score <= 0.0: score = 0.0001
        if score >= 1.0: score = 0.9999
    metrics = _safe_metrics(env.episode_metrics())
    print(f"[END]   Task: {task_name}, Score: {score:.4f}")
    return {"score": score, "metrics": metrics}


def main():
    results = {}
    for task in ("easy", "medium", "hard"):
        results[task] = run_task(task, use_llm=True)

    agg = grade_all_tasks(results)

    # Final safety pass before writing — clamp everything again
    safe_per_task = {}
    for task, s in agg["per_task"].items():
        if s <= 0.0: s = 0.0001
        elif s >= 1.0: s = 0.9999
        else: s = round(s, 4)
        safe_per_task[task] = s

    final_score = agg["final_score"]
    if final_score <= 0.0: final_score = 0.0001
    elif final_score >= 1.0: final_score = 0.9999
    else: final_score = round(final_score, 4)

    with open("results.json", "w") as f:
        json.dump({
            "per_task":    safe_per_task,
            "final_score": final_score,
            "passed":      agg["passed"],
            "details":     {t: _safe_metrics(r["metrics"]) for t, r in results.items()},
        }, f, indent=2)


if __name__ == "__main__":
    main()