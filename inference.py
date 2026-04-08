"""
Inference Script — Customer Support Triage OpenEnv
====================================================
MANDATORY ENV VARS:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.
    OPENENV_BASE_URL  The base URL of the OpenEnv server (your HF Space).

Defaults:
    API_BASE_URL  = "https://router.huggingface.co/v1"
    MODEL_NAME    = "Qwen/Qwen2.5-72B-Instruct"

STDOUT FORMAT (strictly):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
API_KEY      = os.getenv("HF_TOKEN")     or os.getenv("API_KEY") or "dummy_key"

# OpenEnv server URL (your HF Space)
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "https://jiyasadaria2-openenv-support-triage.hf.space")

BENCHMARK = "OpenEnv"
TASKS     = ["single_triage", "queue_triage", "full_resolution"]

# ── Safe imports ───────────────────────────────────────────────────────────────
OpenAI = None
requests = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    pass

try:
    import requests as _req  # type: ignore
    requests = _req
except Exception:
    pass

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support manager AI agent operating in an OpenEnv environment.

Your job is to process customer support tickets step-by-step:
1. CLASSIFY the ticket (category: billing | technical | account | general | security)
2. SET PRIORITY (low | medium | high | urgent)
3. ROUTE to the correct department:
   - billing -> billing_team
   - technical -> engineering
   - account -> account_management
   - general -> general_support
   - security -> security_team
4. RESPOND with a professional, empathetic response (minimum 200 characters)
5. ESCALATE if the issue is severe or complex
6. RESOLVE to close the ticket

Output EXACTLY one valid JSON object per turn. No markdown, no explanation.

Format:
{
  "action_type": "classify" | "set_priority" | "route" | "respond" | "escalate" | "resolve" | "skip",
  "ticket_id": "<id>",
  "category": "<only for classify>",
  "priority": "<only for set_priority>",
  "department": "<only for route>",
  "response_text": "<only for respond, min 200 chars>",
  "resolution_notes": "<optional for resolve>"
}"""


# ── Structured stdout loggers (following official sample pattern) ──────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment HTTP calls ─────────────────────────────────────────────────────
def _call_env(method: str, path: str, **kwargs) -> Dict:
    """Call the OpenEnv HTTP server with retries."""
    if requests is None:
        print("[WARN] requests not available", file=sys.stderr)
        return {}
    url = OPENENV_BASE_URL.rstrip("/") + path
    for attempt in range(5):
        try:
            if method == "GET":
                resp = requests.get(url, timeout=30, **kwargs)
            else:
                resp = requests.post(url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[ENV retry {attempt+1}/5] {e}", file=sys.stderr)
            if attempt < 4:
                time.sleep(2)
    return {}


# ── LLM call ──────────────────────────────────────────────────────────────────
def get_action(client: Any, ticket: Dict, obs: Dict, model: str):
    """Ask the LLM for the next action. Returns (action_dict, error_str_or_None)."""
    user_content = (
        f"Ticket ID: {ticket.get('id')}\n"
        f"Subject: {ticket.get('subject')}\n"
        f"From: {ticket.get('sender')}\n"
        f"Message:\n{ticket.get('body')}\n\n"
        f"Available actions: {', '.join(obs.get('available_actions', []))}\n"
        f"Queue remaining: {obs.get('queue_size', 0)}\n"
        f"Task: {obs.get('task_description', '')}\n\n"
        f"Output ONLY a valid JSON action object."
    )

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.0,
                max_tokens=600,
            )
            raw = (completion.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip()), None
        except Exception as e:
            if attempt == 2:
                return {"action_type": "skip", "ticket_id": ticket.get("id", "0")}, str(e)
            time.sleep(1)

    return {"action_type": "skip", "ticket_id": ticket.get("id", "0")}, "max_retries"


# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(client: Any, task_id: str, model: str) -> None:
    """Run one full episode. Always emits [START] and [END]."""
    log_start(task=task_id, env=BENCHMARK, model=model)

    step_num = 0
    rewards: List[float] = []
    score = 0.0
    success = False

    try:
        reset_resp = _call_env("POST", f"/reset?task_id={task_id}")
        if "session_id" not in reset_resp:
            print(f"[WARN] Reset failed for {task_id}: {reset_resp}", file=sys.stderr)
            return  # finally block will emit [END]

        session_id = reset_resp["session_id"]
        obs = reset_resp.get("observation", {})
        done = False

        while not done:
            step_num += 1
            current = obs.get("current_ticket")
            if not current:
                break

            # Get LLM action (or skip if client unavailable)
            if client is not None:
                action_data, err = get_action(client, current, obs, model)
            else:
                action_data = {"action_type": "skip", "ticket_id": current.get("id", "0")}
                err = "no_client"

            # Send to environment
            step_resp = _call_env(
                "POST",
                f"/step?session_id={session_id}",
                json=action_data,
            )

            reward = float((step_resp.get("reward") or {}).get("total", 0.0))
            done   = bool(step_resp.get("done", True))
            obs    = step_resp.get("observation", {})
            rewards.append(reward)

            act_str = json.dumps(action_data, separators=(",", ":"))
            log_step(step=step_num, action=act_str, reward=reward, done=done, error=err)

        # Grade
        grade_resp = _call_env("POST", f"/grader?session_id={session_id}")
        score   = float(grade_resp.get("score", 0.0))
        success = score >= 0.5

    except Exception as e:
        print(f"[episode error] {e}", file=sys.stderr)

    finally:
        log_end(success=success, steps=step_num, score=score, rewards=rewards)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"API_BASE_URL : {API_BASE_URL}", file=sys.stderr)
    print(f"MODEL_NAME   : {MODEL_NAME}", file=sys.stderr)
    print(f"OPENENV_BASE : {OPENENV_BASE_URL}", file=sys.stderr)

    client = None
    if OpenAI is not None:
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        except Exception as e:
            print(f"[WARN] OpenAI client init failed: {e}", file=sys.stderr)

    for task_id in TASKS:
        try:
            run_episode(client, task_id, MODEL_NAME)
        except Exception as e:
            print(f"[FATAL task={task_id}] {e}", file=sys.stderr)
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[CRASH] {e}", file=sys.stderr)
    sys.exit(0)
