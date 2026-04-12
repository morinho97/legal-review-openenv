"""
server.py — OpenEnv REST API (FastAPI, port 7860)
POST /reset  POST /step  GET /state  POST /validate  GET /health
ALL scores and ratio metrics strictly in open interval (0, 1).
"""
from __future__ import annotations
import os
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from legal_env import (
    LegalReviewEnv, DIFFICULTY_PRESETS,
    grade_episode, rule_based_action, N_ACTIONS, _clamp,
)

app = FastAPI(title="LegalReviewEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_envs: dict[str, LegalReviewEnv] = {}


def _get_env(task: str) -> LegalReviewEnv:
    if task not in DIFFICULTY_PRESETS:
        raise HTTPException(400, f"Unknown task '{task}'. Choose: easy, medium, hard")
    if task not in _envs:
        _envs[task] = LegalReviewEnv(difficulty=task)
    return _envs[task]


def _safe_metrics(m: dict) -> dict:
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


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    task: str = "easy"
    action: int = 0

class ValidateRequest(BaseModel):
    task: str = "easy"


@app.get("/")
def root():
    return {"name": "LegalReviewEnv", "version": "1.0.0", "status": "running",
            "tasks": list(DIFFICULTY_PRESETS.keys()),
            "endpoints": ["POST /reset", "POST /step", "GET /state", "POST /validate", "GET /health"]}

@app.get("/health")
def health():
    return {"status": "ok", "env": "LegalReviewEnv", "version": "1.0.0"}

@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(None)):
    if req is None:
        req = ResetRequest()
    try:
        env = _get_env(req.task)
        obs = env.reset(seed=req.seed)
        return {"observation": obs.tolist(), "task": req.task,
                "n_clauses": DIFFICULTY_PRESETS[req.task]["n_clauses"],
                "obs_dim": len(obs), "status": "reset_ok", "state": env.state()}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"Reset failed: {e}")

@app.post("/step")
def step(req: Optional[StepRequest] = Body(None)):
    if req is None:
        req = StepRequest()
    env = _get_env(req.task)
    if env._done:
        raise HTTPException(400, "Episode done. Call POST /reset first.")
    if not (0 <= req.action < N_ACTIONS):
        raise HTTPException(400, f"Invalid action {req.action}. Must be 0-{N_ACTIONS - 1}.")
    try:
        obs, reward, done, info = env.step(req.action)
        return {"observation": obs.tolist(), "reward": reward, "done": done, "info": info}
    except Exception as e: raise HTTPException(500, f"Step failed: {e}")

@app.get("/state")
def state(task: str = "easy"):
    return _get_env(task).state()

@app.post("/validate")
def validate(req: Optional[ValidateRequest] = Body(None)):
    if req is None:
        req = ValidateRequest()
    try:
        env = LegalReviewEnv(difficulty=req.task)
        env.reset()
        done = False
        while not done:
            s = env.state(); ctx = s.get("current_clause") or {}
            ctx["time_remaining"] = s.get("time_remaining", 999)
            _, _, done, _ = env.step(rule_based_action(ctx))
        score = grade_episode(env)
        # Extra safety clamp
        if score <= 0.0:
            score = 0.0001
        elif score >= 1.0:
            score = 0.9999
        else:
            score = round(score, 4)
            if score <= 0.0: score = 0.0001
            if score >= 1.0: score = 0.9999
        metrics = _safe_metrics(env.episode_metrics())
        return {"task": req.task, "score": score, "metrics": metrics, "status": "validation_ok"}
    except Exception as e:
        raise HTTPException(500, f"Validate failed: {e}")

@app.get("/tasks")
def tasks():
    return {t: {"n_clauses": c["n_clauses"], "contract_type": c["contract_type"],
                "time_budget": c["time_budget"]} for t, c in DIFFICULTY_PRESETS.items()}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)