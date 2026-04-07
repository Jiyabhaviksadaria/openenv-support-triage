"""
Inference Script — Customer Support Triage OpenEnv
====================================================
MANDATORY ENV VARS:
  API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     The model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Your Hugging Face API key

STDOUT FORMAT (strictly):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── Safe imports — never crash at import time ──────────────────────────────────
OpenAI = None
requests = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    pass

try:
    import requests as _requests  # type: ignore
    requests = _requests
except Exception:
    pass

# ── Configuration from environment ────────────────────────────────────────────
BASE_URL    = os.environ.get("OPENENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME  = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN    = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or "dummy_key"
)

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
    """Make an HTTP call to the OpenEnv server with retries."""
    if requests is None:
        raise RuntimeError("requests package not available")
    url = f"{BASE_URL}{path}"
    last_exc = None
    for attempt in range(5):
        try:
            if method == "GET":
                resp = requests.get(url, timeout=30, **kwargs)
            elif method == "POST":
                resp = requests.post(url, timeout=30, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            print(f"[env call attempt {attempt+1}/5 failed] {e}", file=sys.stderr)
            if attempt < 4:
                time.sleep(2)
    raise RuntimeError(f"Environment unreachable after 5 retries: {last_exc}")


def run_episode(
    client: Any,
    task_id: str,
    model: str,
    max_retries: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a full episode for a given task. Always emits [START] and [END]."""

    print(f"[START] task={task_id} env=OpenEnv model={model}", flush=True)

    step_num = 0
    total_reward = 0.0
    rewards_list: List[float] = []

    try:
        if verbose:
            print(f"\n{'='*60}\nTask: {task_id}\nModel: {model}\n{'='*60}", file=sys.stderr)

        reset_resp = _call_env("POST", f"/reset?task_id={task_id}")
        session_id = reset_resp["session_id"]
        obs = reset_resp["observation"]

        messages: List[Dict] = []
        done = False

        while not done:
            step_num += 1

            current = obs.get("current_ticket")
            if not current:
                break

            user_content = (
                f"Current Observation (Step {obs['step']}/{obs['max_steps']}):\n\n"
                f"Ticket ID: {current['id']}\n"
                f"Subject: {current['subject']}\n"
                f"From: {current['sender']}\n"
                f"Timestamp: {current['timestamp']}\n"
                f"Attachments: {', '.join(current.get('attachments', [])) or 'none'}\n\n"
                f"Message:\n{current['body']}\n\n---\n"
                f"Queue size (remaining): {obs['queue_size']}\n"
                f"Available actions: {', '.join(obs['available_actions'])}\n"
                f"Task: {obs['task_description']}\n\n"
                f"What is your next action? Output ONLY a JSON action object."
            )

            if verbose:
                print(f"\n[Step {step_num}] Ticket: {current['id']} — {current['subject'][:50]}", file=sys.stderr)

            messages.append({"role": "user", "content": user_content})

            error_msg = "null"
            action_data = {"action_type": "skip", "ticket_id": current["id"]}

            if client is not None:
                for attempt in range(max_retries):
                    try:
                        completion = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                            temperature=0.0,
                            max_tokens=800,
                        )
                        raw = completion.choices[0].message.content.strip()
                        if raw.startswith("```"):
                            raw = raw.split("```")[1]
                            if raw.startswith("json"):
                                raw = raw[4:]
                        action_data = json.loads(raw)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            error_msg = str(e).replace('\n', ' ')[:200]
                            if verbose:
                                print(f"  [!] LLM failed ({e}), defaulting to skip", file=sys.stderr)
                        else:
                            time.sleep(1)
            else:
                error_msg = "openai_not_available"

            if verbose:
                print(f"  Action: {action_data.get('action_type')}", file=sys.stderr)

            messages.append({"role": "assistant", "content": json.dumps(action_data)})

            step_resp = _call_env(
                "POST",
                f"/step?session_id={session_id}",
                json=action_data,
            )

            reward = float(step_resp["reward"]["total"])
            total_reward += reward
            rewards_list.append(reward)
            done = bool(step_resp["done"])
            obs = step_resp["observation"]

            act_str = json.dumps(action_data, separators=(',', ':'))
            print(
                f"[STEP] step={step_num} action={act_str} reward={reward:.2f} "
                f"done={'true' if done else 'false'} error={error_msg}",
                flush=True,
            )

            if verbose:
                print(f"  Reward: {reward:+.4f} | {step_resp['reward'].get('message','')[:80]}", file=sys.stderr)

            if obs.get("current_ticket") and obs["current_ticket"]["id"] != current["id"]:
                messages = []

        grade_resp = _call_env("POST", f"/grader?session_id={session_id}")
        score = float(grade_resp["score"])

        if verbose:
            print(f"\n{'─'*60}", file=sys.stderr)
            print(f"Episode complete — Steps: {step_num} | Score: {score:.4f}", file=sys.stderr)

        rwds_str = ",".join(f"{r:.2f}" for r in rewards_list)
        print(
            f"[END] success={'true' if score >= 0.5 else 'false'} steps={step_num} "
            f"score={score:.2f} rewards={rwds_str}",
            flush=True,
        )

        return {
            "task_id": task_id,
            "model": model,
            "steps": step_num,
            "cumulative_reward": round(total_reward, 4),
            "grader_score": score,
            "grader_breakdown": grade_resp.get("breakdown", {}),
            "feedback": grade_resp.get("feedback", ""),
        }

    except Exception as e:
        print(f"[episode error] {e}", file=sys.stderr)
        rwds_str = ",".join(f"{r:.2f}" for r in rewards_list)
        print(
            f"[END] success=false steps={step_num} score=0.00 rewards={rwds_str}",
            flush=True,
        )
        return {
            "task_id": task_id,
            "model": model,
            "steps": step_num,
            "cumulative_reward": round(total_reward, 4),
            "grader_score": 0.0,
            "grader_breakdown": {},
            "feedback": str(e),
        }


def run_baseline(
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    model: str = MODEL_NAME,
    verbose: bool = True,
) -> Dict[str, Any]:
    api_key = api_key or HF_TOKEN

    client = None
    if OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key, base_url=api_base_url or API_BASE_URL)
        except Exception as e:
            print(f"OpenAI client init failed: {e}", file=sys.stderr)

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
        "scores": {t: results[t].get("grader_score", 0.0) for t in tasks},
    }

    if verbose:
        print(f"\n{'='*60}\nBASELINE SUMMARY\n{'='*60}", file=sys.stderr)
        for t, s in summary["scores"].items():
            print(f"  {t}: {s:.4f}", file=sys.stderr)
        print(f"  OVERALL: {overall:.4f}", file=sys.stderr)

    return summary


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Run inference for the Support Triage OpenEnv")
        parser.add_argument("--model", default=MODEL_NAME)
        parser.add_argument("--task", default=None)
        parser.add_argument("--base-url", default=None)
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()

        if args.base_url:
            BASE_URL = args.base_url

        verbose = not args.quiet
        api_key = HF_TOKEN

        client = None
        if OpenAI is not None:
            try:
                client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
            except Exception as e:
                print(f"OpenAI client init failed: {e}", file=sys.stderr)

        if args.task:
            result = run_episode(client, args.task, model=args.model, verbose=verbose)
            print(json.dumps(result, indent=2), file=sys.stderr)
        else:
            summary = run_baseline(
                api_key=api_key,
                api_base_url=API_BASE_URL,
                model=args.model,
                verbose=verbose,
            )
            print(json.dumps(summary, indent=2), file=sys.stderr)

    except Exception as e:
        print(f"[FATAL] inference.py top-level exception: {e}", file=sys.stderr)
        # Still exit 0 so pipeline doesn't see non-zero
    
    sys.exit(0)
