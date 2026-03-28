"""
Deterministic graders for each task.
All graders return a score in [0.0, 1.0] and a detailed breakdown dict.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple

from tickets_data import TICKET_BY_ID


# ─── Response Quality Grader ───────────────────────────────────────────────────

def grade_response(response_text: str, ticket: Dict) -> Tuple[float, str]:
    """
    Score a response (0.0–1.0) using keyword coverage and length heuristics.
    Deterministic — no LLM calls.
    """
    if not response_text or len(response_text.strip()) < 20:
        return 0.0, "Response too short or empty"

    response_lower = response_text.lower()
    score = 0.0
    reasons = []

    # 1. Minimum length — a real response needs substance
    length = len(response_text.strip())
    if length >= 200:
        score += 0.25
        reasons.append("good length (+0.25)")
    elif length >= 80:
        score += 0.10
        reasons.append("adequate length (+0.10)")
    else:
        reasons.append("too short (+0.0)")

    # 2. Keyword coverage — does the response address the ticket's key topics?
    expected_keywords = ticket["ground_truth"].get("response_keywords", [])
    if expected_keywords:
        matched = [kw for kw in expected_keywords if kw.lower() in response_lower]
        coverage = len(matched) / len(expected_keywords)
        kw_score = 0.45 * coverage
        score += kw_score
        reasons.append(f"keyword coverage {len(matched)}/{len(expected_keywords)} (+{kw_score:.2f})")

    # 3. Empathy / acknowledgment signals
    empathy_words = ["apologize", "sorry", "understand", "appreciate", "concern"]
    if any(w in response_lower for w in empathy_words):
        score += 0.10
        reasons.append("empathy signal (+0.10)")

    # 4. Action / next-steps commitment
    action_phrases = ["will", "team will", "within", "follow up", "update you", "investigating",
                      "escalat", "priority", "resolve", "fix", "help you"]
    if any(p in response_lower for p in action_phrases):
        score += 0.10
        reasons.append("action commitment (+0.10)")

    # 5. Personalization — mentions ticket subject keywords
    subject_words = ticket["subject"].lower().split()
    significant = [w for w in subject_words if len(w) > 4]
    if significant and any(w in response_lower for w in significant):
        score += 0.10
        reasons.append("personalized (+0.10)")

    final = min(score, 1.0)
    return final, "; ".join(reasons)


# ─── Per-field accuracy helpers ───────────────────────────────────────────────

def _correct(value: Any, expected: Any) -> bool:
    if value is None:
        return False
    return str(value).lower() == str(expected).lower()


# ─── Task 1 Grader: single_triage ─────────────────────────────────────────────

def grade_single_triage(ticket_states: Dict[str, Dict]) -> Dict:
    """
    Task 1 (Easy): Single ticket — classify + set_priority.
    Score = 0.50 * correct_category + 0.50 * correct_priority
    """
    ticket_id = "T005"
    ground_truth = TICKET_BY_ID[ticket_id]["ground_truth"]
    state = ticket_states.get(ticket_id, {})

    cat_correct = _correct(state.get("category"), ground_truth["category"])
    pri_correct = _correct(state.get("priority"), ground_truth["priority"])

    score = 0.5 * int(cat_correct) + 0.5 * int(pri_correct)

    return {
        "score": round(score, 4),
        "breakdown": {
            "category_correct": cat_correct,
            "priority_correct": pri_correct,
            "expected_category": ground_truth["category"],
            "expected_priority": ground_truth["priority"],
            "actual_category": state.get("category"),
            "actual_priority": state.get("priority"),
        },
        "feedback": (
            f"Category: {'✓' if cat_correct else '✗'} "
            f"(got {state.get('category')!r}, expected {ground_truth['category']!r}) | "
            f"Priority: {'✓' if pri_correct else '✗'} "
            f"(got {state.get('priority')!r}, expected {ground_truth['priority']!r})"
        ),
    }


# ─── Task 2 Grader: queue_triage ──────────────────────────────────────────────

def grade_queue_triage(ticket_states: Dict[str, Dict]) -> Dict:
    """
    Task 2 (Medium): Five tickets — classify + prioritize + route each.
    Per-ticket score = category(0.34) + priority(0.33) + routing(0.33)
    Final score = average across all 5 tickets.
    """
    ticket_ids = ["T001", "T004", "T007", "T009", "T011"]
    per_ticket = {}
    total = 0.0

    for tid in ticket_ids:
        gt = TICKET_BY_ID[tid]["ground_truth"]
        state = ticket_states.get(tid, {})

        cat_ok = _correct(state.get("category"), gt["category"])
        pri_ok = _correct(state.get("priority"), gt["priority"])
        dep_ok = _correct(state.get("department"), gt["department"])

        ticket_score = 0.34 * int(cat_ok) + 0.33 * int(pri_ok) + 0.33 * int(dep_ok)
        total += ticket_score

        per_ticket[tid] = {
            "score": round(ticket_score, 4),
            "category_correct": cat_ok,
            "priority_correct": pri_ok,
            "routing_correct": dep_ok,
            "expected": {
                "category": gt["category"],
                "priority": gt["priority"],
                "department": gt["department"],
            },
            "actual": {
                "category": state.get("category"),
                "priority": state.get("priority"),
                "department": state.get("department"),
            },
        }

    final = total / len(ticket_ids)

    return {
        "score": round(final, 4),
        "breakdown": {"per_ticket": per_ticket},
        "feedback": (
            f"Processed {sum(1 for t in per_ticket.values() if t['score'] > 0)}/{len(ticket_ids)} tickets. "
            f"Average score: {final:.2%}"
        ),
    }


# ─── Task 3 Grader: full_resolution ──────────────────────────────────────────

def grade_full_resolution(ticket_states: Dict[str, Dict]) -> Dict:
    """
    Task 3 (Hard): Three complex tickets — full pipeline.
    Per-ticket weights:
      category   0.15
      priority   0.15
      routing    0.15
      response   0.35  (graded by response quality)
      escalation 0.10  (correct escalation decision)
      resolved   0.10

    Final = average across 3 tickets.
    """
    ticket_ids = ["T006", "T010", "T014"]
    per_ticket = {}
    total = 0.0

    for tid in ticket_ids:
        gt = TICKET_BY_ID[tid]["ground_truth"]
        ticket_data = TICKET_BY_ID[tid]
        state = ticket_states.get(tid, {})

        cat_ok = _correct(state.get("category"), gt["category"])
        pri_ok = _correct(state.get("priority"), gt["priority"])
        dep_ok = _correct(state.get("department"), gt["department"])

        # Escalation: correct if (escalated == requires_escalation)
        escalated = state.get("escalated", False)
        escalation_ok = escalated == gt["requires_escalation"]

        resolved = state.get("resolved", False)

        # Response quality
        response_text = state.get("response_text", "") or ""
        resp_score, resp_reason = grade_response(response_text, ticket_data)

        ticket_score = (
            0.15 * int(cat_ok)
            + 0.15 * int(pri_ok)
            + 0.15 * int(dep_ok)
            + 0.35 * resp_score
            + 0.10 * int(escalation_ok)
            + 0.10 * int(resolved)
        )
        total += ticket_score

        per_ticket[tid] = {
            "score": round(ticket_score, 4),
            "category_correct": cat_ok,
            "priority_correct": pri_ok,
            "routing_correct": dep_ok,
            "escalation_correct": escalation_ok,
            "resolved": resolved,
            "response_score": round(resp_score, 4),
            "response_reason": resp_reason,
            "expected": {
                "category": gt["category"],
                "priority": gt["priority"],
                "department": gt["department"],
                "requires_escalation": gt["requires_escalation"],
            },
            "actual": {
                "category": state.get("category"),
                "priority": state.get("priority"),
                "department": state.get("department"),
                "escalated": escalated,
                "resolved": resolved,
            },
        }

    final = total / len(ticket_ids)

    return {
        "score": round(final, 4),
        "breakdown": {"per_ticket": per_ticket},
        "feedback": (
            f"Fully resolved {sum(1 for t in per_ticket.values() if t['resolved'])}/{len(ticket_ids)} tickets. "
            f"Avg response quality: {sum(t['response_score'] for t in per_ticket.values()) / len(ticket_ids):.2%}. "
            f"Final score: {final:.2%}"
        ),
    }


# ─── Dispatcher ───────────────────────────────────────────────────────────────

GRADERS = {
    "single_triage": grade_single_triage,
    "queue_triage": grade_queue_triage,
    "full_resolution": grade_full_resolution,
}


def grade(task_id: str, ticket_states: Dict[str, Dict]) -> Dict:
    """Dispatch to the appropriate grader and return score + breakdown."""
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(GRADERS)}")
    return GRADERS[task_id](ticket_states)
