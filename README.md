---
title: openenv-support-triage
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🎫 Customer Support Triage — OpenEnv Environment
An [OpenEnv](https://openenv.dev)-compliant reinforcement learning environment where AI agents learn to **triage customer support tickets** — a high-value real-world task performed by support teams at scale.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://openenv.dev)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)

---

## 🌍 Motivation

Every company with users processes support tickets. Manual triage is slow, inconsistent, and expensive. Training AI agents on this task has direct commercial value — an agent that correctly classifies, prioritizes, routes, and drafts responses saves hours of human work daily.

This environment gives RL researchers and ML engineers a **standardized, graded benchmark** for ticket triage agents, with reproducible difficulty tiers from simple classification to full end-to-end resolution of complex, ambiguous cases.

---

## 📋 Environment Description

The agent operates on an inbox of customer support tickets. Each ticket is a realistic message containing:
- A subject line and body (varying length and complexity)
- Sender email and timestamp
- Optional attachments (listed by name)

The agent must work through each ticket using a structured action pipeline:

```
classify → set_priority → route → respond → [escalate?] → resolve
```

**Action types:**
| Action | Description | Required field |
|--------|-------------|----------------|
| `classify` | Assign category to ticket | `category` |
| `set_priority` | Set urgency level | `priority` |
| `route` | Send to correct department | `department` |
| `respond` | Draft customer-facing reply | `response_text` |
| `escalate` | Flag for senior attention | — |
| `resolve` | Mark ticket as handled | `resolution_notes` (optional) |
| `skip` | Skip ticket (penalty) | — |

---

## 🔭 Observation Space

```json
{
  "task_id": "string",
  "task_description": "string",
  "step": 0,
  "max_steps": 30,
  "current_ticket": {
    "id": "T006",
    "subject": "PRODUCTION DOWN — all API calls returning 503",
    "body": "...",
    "sender": "oncall@enterprise-client.com",
    "timestamp": "2025-01-06T14:45:00Z",
    "attachments": ["error_logs.txt"]
  },
  "queue_size": 2,
  "processed_count": 0,
  "ticket_states": { "T006": { "category": null, ... } },
  "available_actions": ["classify", "set_priority", "route", "respond", "escalate", "resolve"],
  "done": false
}
```

---

## ⚡ Action Space

```json
{
  "action_type": "classify | set_priority | route | respond | escalate | resolve | skip",
  "ticket_id": "T001",
  "category": "billing | technical | account | general | security",
  "priority": "low | medium | high | urgent",
  "department": "billing_team | engineering | account_management | security_team | general_support",
  "response_text": "Dear customer, ...",
  "resolution_notes": "Resolved via refund"
}
```

---

## 🏆 Tasks

### Task 1: `single_triage` ⭐ Easy
**Objective:** Correctly classify and prioritize a single support ticket.  
**Tickets:** 1 (T005 — CSV export bug)  
**Max steps:** 5  
**Scoring:** `0.5 × category_correct + 0.5 × priority_correct`  
**Expected agent score:** ~0.85

A well-prompted LLM should score ≥ 0.5 easily. Perfect score requires both category and priority correct.

---

### Task 2: `queue_triage` ⭐⭐ Medium
**Objective:** Process a mixed queue of 5 tickets — classify, set priority, and route each correctly.  
**Tickets:** 5 (billing, API feedback, UI bug, locked account, positive review)  
**Max steps:** 20  
**Scoring:** Per-ticket: `0.34 × cat + 0.33 × priority + 0.33 × routing`. Mean across 5 tickets.  
**Expected agent score:** ~0.62

Requires consistent accuracy across tickets with different categories and mixed priorities. Routing requires understanding the billing/engineering/account_management/security_team/general_support split.

---

### Task 3: `full_resolution` ⭐⭐⭐ Hard
**Objective:** Full pipeline on 3 high-stakes, ambiguous tickets requiring expert judgment.  
**Tickets:** T006 (production outage), T010 (security breach), T014 (enterprise SLA dispute)  
**Max steps:** 30  
**Scoring:**
```
Per-ticket = 0.15×category + 0.15×priority + 0.15×routing
           + 0.35×response_quality + 0.10×escalation_decision + 0.10×resolved
```
Response quality is graded by keyword coverage, length, empathy signals, and action commitments.  
**Expected agent score:** ~0.48

The hard task challenges frontier models because:
- All 3 tickets are urgent — misjudging priority tanks the score
- Responses must be substantive (≥200 chars) and cover specific keywords
- Escalation decisions require reading between the lines (all 3 require escalation)
- The grader is deterministic but comprehensive — no free points

---

## 🎁 Reward Function

Rewards are shaped at every step (not just episode end):

| Event | Reward |
|-------|--------|
| Correct category | +0.10 |
| Correct priority | +0.08 |
| Correct routing | +0.08 |
| Response quality (0–1 × 0.15) | 0 to +0.15 |
| Correct escalation decision | +0.05 |
| Resolving ticket | +0.05 |
| Wrong action | −0.05 |
| Unnecessary skip | −0.08 |
| Invalid action | −0.03 |
| Per-step cost | −0.005 |
| All tickets complete (bonus) | +0.15 |

The shaped reward ensures agents get signal throughout the episode, not just at the end. The per-step cost penalizes inefficient agents that take excessive actions.

---

## 📊 Baseline Scores

Model: `gpt-4o-mini` (temperature=0.0)

| Task | Score | Difficulty |
|------|-------|-----------|
| `single_triage` | **0.0000** | Easy |
| `queue_triage` | **0.0000** | Medium |
| `full_resolution` | **0.0000** | Hard |
| **Overall** | **0.0000** | — |

---

## 🚀 Setup & Usage

### Option 1: Docker (Recommended)

```bash
git clone https://huggingface.co/spaces/your-username/openenv-support-triage
cd openenv-support-triage

# Build
docker build -t openenv-support-triage .

# Run
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... openenv-support-triage
```

### Option 2: Local Python

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Run Baseline

```bash
export OPENAI_API_KEY=sk-...
python baseline.py

# Single task only
python baseline.py --task single_triage

# Against deployed space
python baseline.py --base-url https://your-username-openenv-support-triage.hf.space
```

---

## 🔌 API Usage

### 1. Reset (start episode)
```bash
curl -X POST "http://localhost:7860/reset?task_id=single_triage"
```
```json
{
  "session_id": "uuid-here",
  "observation": { "current_ticket": {...}, "available_actions": [...] }
}
```

### 2. Step (take action)
```bash
curl -X POST "http://localhost:7860/step?session_id=uuid-here" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "ticket_id": "T005", "category": "technical"}'
```
```json
{
  "observation": {...},
  "reward": {"total": 0.10, "breakdown": {"category": 0.10}, "message": "Category set to 'technical' — correct ✓"},
  "done": false
}
```

### 3. Grade episode
```bash
curl -X POST "http://localhost:7860/grader?session_id=uuid-here"
```
```json
{
  "score": 1.0,
  "breakdown": {"category_correct": true, "priority_correct": true},
  "feedback": "Category: ✓ | Priority: ✓"
}
```

### 4. List tasks
```bash
curl "http://localhost:7860/tasks"
```

### 5. Run baseline
```bash
curl -X POST "http://localhost:7860/baseline"
```

---

## 📁 Project Structure

```
openenv-support-triage/
├── main.py          # FastAPI server — all OpenEnv endpoints
├── environment.py   # Core env logic: reset/step/state/grade
├── models.py        # Pydantic typed models (Action, Observation, Reward)
├── tasks.py         # Task definitions, metadata, action schemas
├── graders.py       # Deterministic graders for all 3 tasks
├── tickets_data.py  # 15 realistic tickets with ground truth labels
├── baseline.py      # Baseline inference script (OpenAI API)
├── openenv.yaml     # OpenEnv spec metadata
├── Dockerfile       # HF Spaces compatible container
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

```python
import requests

# Full episode example
BASE = "http://localhost:7860"

# Start episode
resp = requests.post(f"{BASE}/reset?task_id=single_triage").json()
session_id = resp["session_id"]
ticket_id = resp["observation"]["current_ticket"]["id"]

# Classify
requests.post(f"{BASE}/step?session_id={session_id}", json={
    "action_type": "classify", "ticket_id": ticket_id, "category": "technical"
})

# Set priority
requests.post(f"{BASE}/step?session_id={session_id}", json={
    "action_type": "set_priority", "ticket_id": ticket_id, "priority": "medium"
})

# Grade
score = requests.post(f"{BASE}/grader?session_id={session_id}").json()
print(f"Score: {score['score']}")  # → 1.0
```

---

## 📜 License

MIT License — see LICENSE file.
