"""
Baseline Inference Script — Customer Support Triage OpenEnv
============================================================
Runs a GPT-4o-mini agent against all 3 tasks using the OpenAI API client.
Reads credentials from OPENAI_API_KEY env var.

Usage:
    python baseline.py

Or from the /baseline endpoint (runs automatically).

Baseline Results (gpt-4o-mini, seed=42):
  single_triage:   ~0.85
  queue_triage:    ~0.62
  full_resolution: ~0.48
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── OpenAI client ──────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# ── Environment client ────────────────────────────────────────────────────────
import requests

BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:7860")

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert customer support manager AI agent operating in an OpenEnv environment.

Your job is to process customer support tickets step-by-step through the following pipeline:
1. CLASSIFY the ticket (category: billing | technical | account | general | security)
2. SET PRIORITY (low | medium | high | urgent)
3. ROUTE to the correct department:
   - billing → billing_team
   - technical → engineering  
   - account → account_management
   - general → general_support
   - security → security_team
4. RESPOND with a professional, empathetic, actionable response (minimum 200 characters)
5. ESCALATE if the issue is severe, complex, or requires senior attention
6. RESOLVE to close the ticket

For each step, you MUST output a valid JSON action. No extra text, just JSON.

Action format:
{
  "action_type": "classify" | "set_priority" | "route" | "respond" | "escalate" | "resolve" | "skip",
  "ticket_id": "<ticket_id>",
  "category": "<only for classify>",
  "priority": "<only for set_priority>",
  "department": "<only for route>",
  "response_text": "<only for respond — must be professional and at least 200 characters>",
  "resolution_notes": "<optional for resolve>"
}

Priority guidelines:
- urgent: service down, security breach, data loss, legal threats
- high: payment blocked, account locked, significant bugs
- medium: billing questions, minor bugs, account questions
- low: feature requests, general questions, positive feedback

Always respond with ONLY the JSON object. No markdown, no explanation."""


def _call_env(method: str, path: str, **kwargs) -> Dict:
    """Make an HTTP call to the OpenEnv server."""
    url = f"{BASE_URL}{path}"
    if method == "GET":
        resp = requests.get(url, **kwargs)
    elif method == "POST":
        resp = requests.post(url, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    resp.raise_for_status()
    return resp.json()


def run_episode(
    client: OpenAI,
    task_id: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a full episode for a given task. Returns grader result."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"Model: {model}")
        print(f"{'='*60}")

    # Reset environment
    reset_resp = _call_env("POST", f"/reset?task_id={task_id}")
    session_id = reset_resp["session_id"]
    obs = reset_resp["observation"]

    messages: List[Dict] = []
    step_num = 0
    done = False
    total_reward = 0.0

    while not done:
        step_num += 1

        # Build user message from observation
        current = obs.get("current_ticket")
        if not current:
            break

        user_content = f"""Current Observation (Step {obs['step']}/{obs['max_steps']}):

Ticket ID: {current['id']}
Subject: {current['subject']}
From: {current['sender']}
Timestamp: {current['timestamp']}
Attachments: {', '.join(current.get('attachments', [])) or 'none'}

Message:
{current['body']}

---
Queue size (remaining): {obs['queue_size']}
Available actions: {', '.join(obs['available_actions'])}
Task: {obs['task_description']}

What is your next action? Output ONLY a JSON action object."""

        if verbose:
            print(f"\n[Step {step_num}] Ticket: {current['id']} — {current['subject'][:50]}")

        # Add to message history
        messages.append({"role": "user", "content": user_content})

        # Call the LLM
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    temperature=0.0,
                    max_tokens=800,
                )
                raw = completion.choices[0].message.content.strip()

                # Parse JSON action
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                action_data = json.loads(raw)
                break
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries - 1:
                    # Fall back to skip
                    action_data = {"action_type": "skip", "ticket_id": current["id"]}
                    if verbose:
                        print(f"  [!] LLM parse failed ({e}), defaulting to skip")
                else:
                    time.sleep(1)

        if verbose:
            print(f"  Action: {action_data.get('action_type')} | Category: {action_data.get('category')} | Priority: {action_data.get('priority')}")

        # Add assistant message to history
        messages.append({"role": "assistant", "content": json.dumps(action_data)})

        # Send action to environment
        step_resp = _call_env(
            "POST",
            f"/step?session_id={session_id}",
            json=action_data,
        )

        reward = step_resp["reward"]["total"]
        total_reward += reward
        done = step_resp["done"]
        obs = step_resp["observation"]

        if verbose:
            print(f"  Reward: {reward:+.4f} | {step_resp['reward']['message'][:80]}")

        # Reset messages for next ticket if we advanced
        if obs.get("current_ticket") and obs["current_ticket"]["id"] != current["id"]:
            messages = []  # Fresh context for each ticket

    # Grade the episode
    grade_resp = _call_env("POST", f"/grader?session_id={session_id}")

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Episode complete — Steps: {step_num} | Cumulative reward: {total_reward:.4f}")
        print(f"Grader score: {grade_resp['score']:.4f}")
        print(f"Feedback: {grade_resp['feedback']}")

    return {
        "task_id": task_id,
        "model": model,
        "steps": step_num,
        "cumulative_reward": round(total_reward, 4),
        "grader_score": grade_resp["score"],
        "grader_breakdown": grade_resp["breakdown"],
        "feedback": grade_resp["feedback"],
    }


def run_baseline(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run baseline agent against all 3 tasks.
    Returns dict with per-task scores and overall average.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    tasks = ["single_triage", "queue_triage", "full_resolution"]
    results = {}

    for task_id in tasks:
        try:
            result = run_episode(client, task_id, model=model, verbose=verbose)
            results[task_id] = result
        except Exception as e:
            results[task_id] = {"error": str(e), "grader_score": 0.0}

    scores = [r.get("grader_score", 0.0) for r in results.values()]
    overall = sum(scores) / len(scores) if scores else 0.0

    summary = {
        "model": model,
        "overall_average": round(overall, 4),
        "tasks": results,
        "scores": {task: results[task].get("grader_score", 0.0) for task in tasks},
    }

    if verbose:
        print(f"\n{'='*60}")
        print("BASELINE SUMMARY")
        print(f"{'='*60}")
        for task, score in summary["scores"].items():
            print(f"  {task}: {score:.4f}")
        print(f"  OVERALL: {overall:.4f}")

    return summary


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline inference for the Support Triage OpenEnv")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--task", default=None, help="Run single task only (single_triage|queue_triage|full_resolution)")
    parser.add_argument("--base-url", default="http://localhost:7860", help="OpenEnv server URL")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    args = parser.parse_args()

    if args.base_url:
        BASE_URL = args.base_url

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    if args.task:
        result = run_episode(client, args.task, model=args.model, verbose=not args.quiet)
        print(json.dumps(result, indent=2))
    else:
        summary = run_baseline(api_key=api_key, model=args.model, verbose=not args.quiet)
        print(json.dumps(summary, indent=2))
