"""
Task definitions for the Customer Support Triage environment.
Each task defines: objective, ticket set, max_steps, required_actions, and reward weights.
"""
from __future__ import annotations
from typing import Dict, List, Any

from tickets_data import TASK_TICKETS

# ─── Action Schema (shared across all tasks) ──────────────────────────────────

ACTION_SCHEMA = {
    "action_type": {
        "type": "string",
        "required": True,
        "enum": ["classify", "set_priority", "route", "respond", "escalate", "resolve", "skip"],
        "description": "The type of action to perform on the ticket",
    },
    "ticket_id": {
        "type": "string",
        "required": True,
        "description": "ID of the ticket to act on (e.g. 'T005')",
    },
    "category": {
        "type": "string",
        "required": "when action_type == 'classify'",
        "enum": ["billing", "technical", "account", "general", "security"],
    },
    "priority": {
        "type": "string",
        "required": "when action_type == 'set_priority'",
        "enum": ["low", "medium", "high", "urgent"],
    },
    "department": {
        "type": "string",
        "required": "when action_type == 'route'",
        "enum": ["billing_team", "engineering", "account_management", "security_team", "general_support"],
    },
    "response_text": {
        "type": "string",
        "required": "when action_type == 'respond'",
        "description": "Full text of the customer response to send",
    },
    "resolution_notes": {
        "type": "string",
        "required": False,
        "description": "Optional notes when resolving a ticket",
    },
}

# ─── Task Definitions ─────────────────────────────────────────────────────────

TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "single_triage",
        "name": "Single Ticket Triage",
        "difficulty": "easy",
        "description": (
            "A single support ticket has arrived. Your objective is to correctly "
            "classify it by category and set the appropriate priority level."
        ),
        "objective": (
            "For the presented ticket:\n"
            "1. Use 'classify' action to set the correct category "
            "(billing/technical/account/general/security)\n"
            "2. Use 'set_priority' action to assign the correct priority "
            "(low/medium/high/urgent)\n"
            "Episode ends after both actions are taken."
        ),
        "ticket_ids": TASK_TICKETS["single_triage"],
        "max_steps": 5,
        "required_actions": ["classify", "set_priority"],
        "reward_weights": {
            "correct_category": 0.50,
            "correct_priority": 0.50,
        },
        "action_schema": ACTION_SCHEMA,
        "scoring_rubric": (
            "Score = 0.5 * (category_correct) + 0.5 * (priority_correct). "
            "Perfect score = 1.0 requires both correct."
        ),
    },
    {
        "task_id": "queue_triage",
        "name": "Queue Triage (5 Tickets)",
        "difficulty": "medium",
        "description": (
            "A queue of 5 support tickets with varying complexity has arrived. "
            "Process each ticket by classifying it, setting priority, and routing "
            "it to the correct department. Order and efficiency matter."
        ),
        "objective": (
            "For each of the 5 tickets in the queue:\n"
            "1. 'classify' — assign the correct category\n"
            "2. 'set_priority' — assign urgency level\n"
            "3. 'route' — send to the correct department\n\n"
            "Tickets: T001 (billing issue), T004 (API feedback), T007 (UI bug), "
            "T009 (account locked), T011 (positive feedback)\n\n"
            "Score is averaged across all 5 tickets."
        ),
        "ticket_ids": TASK_TICKETS["queue_triage"],
        "max_steps": 20,
        "required_actions": ["classify", "set_priority", "route"],
        "reward_weights": {
            "correct_category": 0.34,
            "correct_priority": 0.33,
            "correct_routing": 0.33,
        },
        "action_schema": ACTION_SCHEMA,
        "scoring_rubric": (
            "Per-ticket score = 0.34 * cat_correct + 0.33 * priority_correct + 0.33 * routing_correct. "
            "Final score = mean across all 5 tickets."
        ),
    },
    {
        "task_id": "full_resolution",
        "name": "Full Resolution Pipeline (3 Complex Tickets)",
        "difficulty": "hard",
        "description": (
            "Three high-stakes, ambiguous support tickets requiring expert handling. "
            "You must classify, prioritize, route, compose a professional response, "
            "decide whether to escalate, and mark as resolved. "
            "Tickets involve: a production outage, a security breach report, and a "
            "breaking API change dispute from an enterprise customer."
        ),
        "objective": (
            "For each of the 3 complex tickets (T006, T010, T014):\n"
            "1. 'classify' — correct category\n"
            "2. 'set_priority' — correct urgency (hint: all are urgent)\n"
            "3. 'route' — correct department\n"
            "4. 'respond' — write a professional, empathetic, actionable response (≥200 chars)\n"
            "5. 'escalate' — if the ticket requires senior/specialist involvement\n"
            "6. 'resolve' — mark ticket as handled\n\n"
            "Response quality is graded on length, keyword coverage, empathy, and action commitment."
        ),
        "ticket_ids": TASK_TICKETS["full_resolution"],
        "max_steps": 30,
        "required_actions": ["classify", "set_priority", "route", "respond", "resolve"],
        "optional_actions": ["escalate"],
        "reward_weights": {
            "correct_category": 0.15,
            "correct_priority": 0.15,
            "correct_routing": 0.15,
            "response_quality": 0.35,
            "escalation_decision": 0.10,
            "resolved": 0.10,
        },
        "action_schema": ACTION_SCHEMA,
        "scoring_rubric": (
            "Per-ticket: 0.15*cat + 0.15*priority + 0.15*routing + 0.35*response_quality "
            "+ 0.10*escalation_correct + 0.10*resolved. Final = mean of 3 tickets."
        ),
    },
]

TASKS_BY_ID: Dict[str, Dict] = {t["task_id"]: t for t in TASKS}
