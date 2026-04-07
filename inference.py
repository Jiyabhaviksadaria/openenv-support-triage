from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── SET THESE (IMPORTANT) ─────────────────────────────────────────────────────
BASE_URL = os.environ.get(
    "OPENENV_BASE_URL",
    "https://jiyasadaria2-openenv-support-triage.hf.space"
)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

HF_TOKEN = (
    os.environ.get("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")
    or os.environ.get("API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or "dummy_key"
)

print("DEBUG BASE_URL:", BASE_URL, file=sys.stderr)
print("DEBUG TOKEN:", HF_TOKEN[:10], file=sys.stderr)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except:
    OpenAI = None

try:
    import requests
except:
    requests = None

# ── Safe API call ─────────────────────────────────────────────────────────────
def _call_env(method: str, path: str, **kwargs) -> Dict:
    if requests is None:
        return {"error": "requests_missing"}

    if not BASE_URL:
        return {"error": "missing_base_url"}

    url = BASE_URL + path

    for attempt in range(3):
        try:
            if method == "GET":
                res = requests.get(url, timeout=15)
            else:
                res = requests.post(url, timeout=15, **kwargs)

            return res.json()

        except Exception as e:
            print(f"[ENV RETRY {attempt+1}] {e}", file=sys.stderr)
            time.sleep(1)

    return {"error": "env_failed"}

# ── Main logic ────────────────────────────────────────────────────────────────
def run_episode(client: Any, task_id: str, model: str):

    print(f"[START] task={task_id} env=OpenEnv model={model}", flush=True)

    try:
        reset_resp = _call_env("POST", f"/reset?task_id={task_id}")

        if "session_id" not in reset_resp:
            print("[RESET ERROR]", reset_resp, file=sys.stderr)
            print("[END] success=false steps=0 score=0 rewards=", flush=True)
            return

        session_id = reset_resp["session_id"]
        obs = reset_resp.get("observation", {})

        step_num = 0

        while True:
            step_num += 1

            current = obs.get("current_ticket")
            if not current:
                break

            action_data = {
                "action_type": "skip",
                "ticket_id": current.get("id", "0")
            }

            error_msg = "null"

            # ── LLM CALL ─────────────────────────────
            if client:
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Return valid JSON"}],
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

            # ── STEP CALL ────────────────────────────
            step_resp = _call_env(
                "POST",
                f"/step?session_id={session_id}",
                json=action_data,
            )

            reward = float(step_resp.get("reward", {}).get("total", 0))
            done = bool(step_resp.get("done", True))
            obs = step_resp.get("observation", {})

            print(
                f"[STEP] step={step_num} action={json.dumps(action_data)} "
                f"reward={reward:.2f} done={'true' if done else 'false'} error={error_msg}",
                flush=True,
            )

            if done:
                break

        # ── GRADER ────────────────────────────────
        grade_resp = _call_env("POST", f"/grader?session_id={session_id}")
        score = float(grade_resp.get("score", 0.0))

        print(
            f"[END] success={'true' if score >= 0.5 else 'false'} "
            f"steps={step_num} score={score:.2f} rewards=0.00",
            flush=True,
        )

    except Exception as e:
        import traceback
        print("[FATAL ERROR]", e, file=sys.stderr)
        traceback.print_exc()

        print("[END] success=false steps=0 score=0 rewards=", flush=True)

# ── Runner ───────────────────────────────────────────────────────────────────
def run():
    client = None

    if OpenAI:
        try:
            client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        except Exception as e:
            print("[CLIENT ERROR]", e, file=sys.stderr)

    tasks = ["single_triage", "queue_triage", "full_resolution"]

    for t in tasks:
        run_episode(client, t, MODEL_NAME)

# ── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("[CRASH PREVENTED]", e, file=sys.stderr)
        print("[END] success=false steps=0 score=0 rewards=", flush=True)

    sys.exit(0)
