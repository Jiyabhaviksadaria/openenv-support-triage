from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional


# ── Safe imports ──────────────────────────────────────────────────────────────
OpenAI = None
requests = None

try:
    from openai import OpenAI
except Exception:
    pass

try:
    import requests as _requests
    requests = _requests
except Exception:
    pass

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:7860")

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://router.huggingface.co/v1"
)

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "Qwen/Qwen2.5-72B-Instruct"
)

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy_key"

print("DEBUG API:", API_BASE_URL, file=sys.stderr)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support manager AI agent.

Always respond ONLY with valid JSON.
"""

# ── Safe API call ─────────────────────────────────────────────────────────────
def _call_env(method: str, path: str, **kwargs) -> Dict:
    if requests is None:
        print("[ERROR] requests not installed", file=sys.stderr)
        return {"error": "requests_not_available"}

    url = f"{BASE_URL}{path}"

    for attempt in range(3):
        try:
            if method == "GET":
                resp = requests.get(url, timeout=20, **kwargs)
            else:
                resp = requests.post(url, timeout=20, **kwargs)

            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            print(f"[Retry {attempt+1}] {e}", file=sys.stderr)
            time.sleep(1)

    return {"error": "env_failed"}

# ── Main logic ────────────────────────────────────────────────────────────────
def run_episode(client: Any, task_id: str, model: str) -> Dict:

    print(f"[START] task={task_id} env=OpenEnv model={model}", flush=True)

    try:
        reset_resp = _call_env("POST", f"/reset?task_id={task_id}")

        if "session_id" not in reset_resp:
            raise RuntimeError(f"Bad response: {reset_resp}")

        session_id = reset_resp["session_id"]
        obs = reset_resp["observation"]

        done = False
        step_num = 0

        while not done:
            step_num += 1

            current = obs.get("current_ticket")
            if not current:
                break

            action_data = {
                "action_type": "skip",
                "ticket_id": current["id"]
            }

            error_msg = "null"

            # ── LLM CALL ─────────────────────────────
            if client:
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Respond in JSON"}],
                        max_tokens=200
                    )

                    raw = completion.choices[0].message.content.strip()

                    try:
                        action_data = json.loads(raw)
                    except:
                        print("[JSON ERROR]", raw, file=sys.stderr)

                except Exception as e:
                    error_msg = str(e)

            else:
                error_msg = "no_client"

            # ── Step call ────────────────────────────
            step_resp = _call_env(
                "POST",
                f"/step?session_id={session_id}",
                json=action_data,
            )

            reward = float(step_resp.get("reward", {}).get("total", 0))
            done = bool(step_resp.get("done", True))
            obs = step_resp.get("observation", {})

            done_str = "true" if done else "false"
            act_str = json.dumps(action_data, separators=(',', ':'))
            print(
                f"[STEP] step={step_num} action={act_str} "
                f"reward={reward:.2f} done={done_str} error={error_msg}",
                flush=True,
            )

        # Grade the episode
        grade_resp = _call_env("POST", f"/grader?session_id={session_id}")
        score = float(grade_resp.get("score", 0.0))
        success_str = "true" if score >= 0.5 else "false"

        print(
            f"[END] success={success_str} steps={step_num} score={score:.2f} rewards=0.00",
            flush=True,
        )

        return {"status": "done", "score": score}

    except Exception as e:
        import traceback
        print("[ERROR]", e, file=sys.stderr)
        traceback.print_exc()

        print(
            f"[END] success=false steps=0 score=0.0 rewards=",
            flush=True,
        )

        return {"error": str(e)}

# ── Runner ───────────────────────────────────────────────────────────────────
def run():
    client = None

    if OpenAI:
        try:
            client = OpenAI(
                api_key=HF_TOKEN,
                base_url=API_BASE_URL
            )
        except Exception as e:
            print("Client init failed:", e, file=sys.stderr)

    tasks = ["single_triage", "queue_triage", "full_resolution"]

    for t in tasks:
        run_episode(client, t, MODEL_NAME)

# ── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
    sys.exit(0)