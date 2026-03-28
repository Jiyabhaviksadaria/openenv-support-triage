"""
Customer Support Triage — OpenEnv FastAPI Server

Endpoints:
  POST /reset          → start new episode
  POST /step           → take action
  GET  /state          → current internal state
  GET  /tasks          → list tasks + action schema
  POST /grader         → score completed episode
  POST /baseline       → run baseline inference
  GET  /health         → health check
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from models import Action, GraderResponse, ResetResponse, StateResponse, StepResponse
from environment import SupportTriageEnv
from tasks import TASKS, TASKS_BY_ID

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Support Triage — OpenEnv",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to triage "
        "customer support tickets: classify, prioritize, route, respond, and resolve."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store (in-memory, TTL = 2 hours) ──────────────────────────────────

SESSIONS: Dict[str, Dict] = {}
SESSION_TTL = timedelta(hours=2)


def _get_session(session_id: str) -> SupportTriageEnv:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    entry = SESSIONS[session_id]
    if datetime.utcnow() - entry["created_at"] > SESSION_TTL:
        del SESSIONS[session_id]
        raise HTTPException(status_code=410, detail="Session expired. Call /reset to start a new episode.")
    return entry["env"]


def _cleanup_sessions():
    expired = [
        sid for sid, entry in SESSIONS.items()
        if datetime.utcnow() - entry["created_at"] > SESSION_TTL
    ]
    for sid in expired:
        del SESSIONS[sid]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint — required by real OpenEnv spec."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # Basic ACK to maintain the connection for spec validators
            await websocket.send_json({"status": "ok"})
    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health():
    """Health check — returns 200 if the service is running."""
    return {"status": "ok", "active_sessions": len(SESSIONS), "tasks": len(TASKS)}


@app.get("/")
async def root():
    return {
        "name": "Customer Support Triage OpenEnv",
        "version": "1.0.0",
        "description": (
            "An RL environment for training and evaluating agents on customer support triage. "
            "Agents must classify tickets, set priorities, route them, and draft responses."
        ),
        "tasks": [t["task_id"] for t in TASKS],
        "endpoints": {
            "reset": "POST /reset?task_id=single_triage",
            "step": "POST /step?session_id=<id>",
            "state": "GET /state?session_id=<id>",
            "tasks": "GET /tasks",
            "grader": "POST /grader?session_id=<id>",
            "baseline": "POST /baseline",
        },
    }


class ResetRequest(BaseModel):
    task_id: str = "single_triage"
    seed: Optional[int] = None

@app.post("/reset", response_model=ResetResponse)
async def reset(
    request: Optional[ResetRequest] = None,
    task_id: Optional[str] = Query(default=None, description="Task to run"),
    seed: Optional[int] = Query(default=None, description="Random seed"),
):
    """
    Start a new episode. Returns a session_id and initial observation.
    Pass session_id to /step and /state.
    """
    actual_task_id = (request and request.task_id) or task_id or "single_triage"
    actual_seed = (request and request.seed) or seed

    if actual_task_id not in TASKS_BY_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {actual_task_id!r}. Valid options: {list(TASKS_BY_ID.keys())}"
        )
    _cleanup_sessions()

    session_id = str(uuid.uuid4())
    env = SupportTriageEnv(task_id=actual_task_id, seed=actual_seed)
    obs = env.reset()

    SESSIONS[session_id] = {
        "env": env,
        "task_id": actual_task_id,
        "created_at": datetime.utcnow(),
    }

    return ResetResponse(
        session_id=session_id,
        observation=obs,
        task_id=actual_task_id,
        task_description=TASKS_BY_ID[actual_task_id]["description"],
    )


@app.post("/step", response_model=StepResponse)
async def step(
    action: Action,
    session_id: str = Query(..., description="Session ID from /reset"),
):
    """
    Take one action in the environment.
    Returns observation, reward, done flag, and info dict.
    """
    env = _get_session(session_id)
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
async def state(
    session_id: str = Query(..., description="Session ID from /reset"),
):
    """
    Return the full internal environment state (for debugging / introspection).
    Includes full ticket queue, all ticket states, and trajectory so far.
    """
    env = _get_session(session_id)
    s = env.state()
    s.session_id = session_id
    return s


@app.get("/tasks")
async def get_tasks():
    """
    List all available tasks with their descriptions, difficulty, and action schema.
    """
    return {
        "tasks": [
            {
                "task_id": t["task_id"],
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "objective": t["objective"],
                "max_steps": t["max_steps"],
                "ticket_count": len(t["ticket_ids"]),
                "required_actions": t["required_actions"],
                "reward_weights": t["reward_weights"],
                "scoring_rubric": t["scoring_rubric"],
                "action_schema": t["action_schema"],
            }
            for t in TASKS
        ]
    }


@app.post("/grader", response_model=GraderResponse)
async def grader(
    session_id: str = Query(..., description="Session ID from /reset"),
):
    """
    Grade the completed (or in-progress) episode.
    Returns score in [0.0, 1.0] with detailed breakdown.
    """
    env = _get_session(session_id)
    result = env.grade()
    return GraderResponse(
        session_id=session_id,
        task_id=env.task_id,
        score=result["score"],
        breakdown=result["breakdown"],
        feedback=result["feedback"],
    )


@app.post("/baseline")
async def baseline():
    """
    Run the baseline inference script against all 3 tasks using the OpenAI API.
    Requires OPENAI_API_KEY environment variable.
    Returns baseline scores for all tasks.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY environment variable not set. Cannot run baseline."
        )

    try:
        from baseline import run_baseline
        results = run_baseline(api_key=api_key)
        return {"status": "completed", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {str(e)}")


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
