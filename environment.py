"""
Customer Support Triage Environment — Core Logic.

Implements the OpenEnv interface:
  reset()  → initial Observation
  step()   → (Observation, Reward, done, info)
  state()  → StateResponse (full internal state)
  grade()  → final episode score
"""
from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action, ActionType, Observation, Reward, StateResponse, Ticket, TicketState
)
from tickets_data import TICKET_BY_ID, TASK_TICKETS
from tasks import TASKS_BY_ID
import graders as grader_module


def _make_ticket(tid: str) -> Ticket:
    raw = TICKET_BY_ID[tid]
    return Ticket(
        id=raw["id"],
        subject=raw["subject"],
        body=raw["body"],
        sender=raw["sender"],
        timestamp=raw["timestamp"],
        attachments=raw.get("attachments", []),
    )


class SupportTriageEnv:
    """
    Stateful environment for a single episode.
    Each call to reset() creates a fresh episode for the given task.
    """

    # ── Step-level reward constants ────────────────────────────────────────────
    REWARD_CORRECT_CATEGORY = 0.10
    REWARD_CORRECT_PRIORITY = 0.08
    REWARD_CORRECT_ROUTING = 0.08
    REWARD_ESCALATION_CORRECT = 0.05
    REWARD_RESOLVED = 0.05
    PENALTY_WRONG_ACTION = -0.05
    PENALTY_SKIP = -0.08
    PENALTY_PER_STEP = -0.005     # efficiency pressure
    PENALTY_INVALID_ACTION = -0.03
    BONUS_ALL_COMPLETE = 0.15     # bonus for completing all tickets

    def __init__(self, task_id: str = "single_triage", seed: Optional[int] = None):
        if task_id not in TASKS_BY_ID:
            raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(TASKS_BY_ID)}")
        self.task_id = task_id
        self.task = TASKS_BY_ID[task_id]
        self.seed = seed
        self._initialized = False

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        ticket_ids = self.task["ticket_ids"]
        self.ticket_queue: List[Ticket] = [_make_ticket(tid) for tid in ticket_ids]
        self.current_ticket: Optional[Ticket] = self.ticket_queue.pop(0) if self.ticket_queue else None

        # Keyed by ticket_id
        self.ticket_states: Dict[str, Dict] = {
            tid: {
                "ticket_id": tid,
                "category": None,
                "priority": None,
                "department": None,
                "response_text": None,
                "escalated": False,
                "resolved": False,
                "actions_taken": [],
            }
            for tid in ticket_ids
        }

        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.trajectory: List[Dict] = []
        self._initialized = True
        return self._make_observation()

    # ── step ───────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")
        if self.done:
            obs = self._make_observation()
            return obs, Reward(total=0.0, message="Episode already done"), True, {}

        self.step_count += 1
        reward_breakdown: Dict[str, float] = {}
        messages: List[str] = []

        # ── Validate action ────────────────────────────────────────────────────
        ticket_id = action.ticket_id

        # Does the ticket exist in this task?
        if ticket_id not in self.ticket_states:
            r = Reward(
                total=self.PENALTY_INVALID_ACTION,
                breakdown={"invalid_ticket": self.PENALTY_INVALID_ACTION},
                message=f"Ticket {ticket_id!r} is not in this task's ticket set.",
            )
            self.total_reward += r.total
            obs = self._make_observation()
            return obs, r, self.done, {"error": r.message}

        # Is this the ticket we should be working on?
        if self.current_ticket and ticket_id != self.current_ticket.id:
            r = Reward(
                total=self.PENALTY_INVALID_ACTION,
                breakdown={"wrong_ticket": self.PENALTY_INVALID_ACTION},
                message=f"Must act on current ticket {self.current_ticket.id!r}, not {ticket_id!r}.",
            )
            self.total_reward += r.total
            obs = self._make_observation()
            return obs, r, self.done, {"error": r.message}

        ts = self.ticket_states[ticket_id]
        gt = TICKET_BY_ID[ticket_id]["ground_truth"]

        # ── Apply step penalty ─────────────────────────────────────────────────
        reward_breakdown["step_cost"] = self.PENALTY_PER_STEP

        # ── Dispatch action ────────────────────────────────────────────────────
        if action.action_type == ActionType.classify:
            if action.category is None:
                return self._invalid("`category` is required for classify action")
            ts["actions_taken"].append("classify")
            ts["category"] = action.category.value
            correct = (action.category.value == gt["category"])
            val = self.REWARD_CORRECT_CATEGORY if correct else self.PENALTY_WRONG_ACTION
            reward_breakdown["category"] = val
            messages.append(
                f"Category set to {action.category.value!r} — "
                f"{'correct ✓' if correct else 'incorrect ✗ (expected ' + gt['category'] + ')'}"
            )

        elif action.action_type == ActionType.set_priority:
            if action.priority is None:
                return self._invalid("`priority` is required for set_priority action")
            ts["actions_taken"].append("set_priority")
            ts["priority"] = action.priority.value
            correct = (action.priority.value == gt["priority"])
            val = self.REWARD_CORRECT_PRIORITY if correct else self.PENALTY_WRONG_ACTION
            reward_breakdown["priority"] = val
            messages.append(
                f"Priority set to {action.priority.value!r} — "
                f"{'correct ✓' if correct else 'incorrect ✗ (expected ' + gt['priority'] + ')'}"
            )

        elif action.action_type == ActionType.route:
            if action.department is None:
                return self._invalid("`department` is required for route action")
            ts["actions_taken"].append("route")
            ts["department"] = action.department.value
            correct = (action.department.value == gt["department"])
            val = self.REWARD_CORRECT_ROUTING if correct else self.PENALTY_WRONG_ACTION
            reward_breakdown["routing"] = val
            messages.append(
                f"Routed to {action.department.value!r} — "
                f"{'correct ✓' if correct else 'incorrect ✗ (expected ' + gt['department'] + ')'}"
            )

        elif action.action_type == ActionType.respond:
            if not action.response_text:
                return self._invalid("`response_text` is required for respond action")
            ts["actions_taken"].append("respond")
            ts["response_text"] = action.response_text
            resp_score, resp_reason = grader_module.grade_response(
                action.response_text, TICKET_BY_ID[ticket_id]
            )
            # Scale response reward: up to 0.15 per step
            val = 0.15 * resp_score
            reward_breakdown["response_quality"] = val
            messages.append(f"Response recorded. Quality score: {resp_score:.2f} — {resp_reason}")

        elif action.action_type == ActionType.escalate:
            ts["actions_taken"].append("escalate")
            ts["escalated"] = True
            if gt["requires_escalation"]:
                reward_breakdown["escalation"] = self.REWARD_ESCALATION_CORRECT
                messages.append("Escalated ✓ (this ticket required escalation)")
            else:
                reward_breakdown["escalation"] = self.PENALTY_WRONG_ACTION
                messages.append("Escalated ✗ (this ticket did NOT require escalation — unnecessary escalation)")

        elif action.action_type == ActionType.resolve:
            ts["actions_taken"].append("resolve")
            ts["resolved"] = True
            ts["resolution_notes"] = action.resolution_notes or ""

            # Grant resolve reward only if minimum actions were taken
            required = self.task.get("required_actions", [])
            taken = set(ts["actions_taken"])
            if all(r in taken for r in required if r != "resolve"):
                reward_breakdown["resolved"] = self.REWARD_RESOLVED
                messages.append("Ticket resolved ✓")
            else:
                missing = [r for r in required if r not in taken and r != "resolve"]
                reward_breakdown["resolved"] = 0.0
                messages.append(f"Ticket resolved but missing actions: {missing}")

            # Advance to next ticket
            self._advance_ticket()

        elif action.action_type == ActionType.skip:
            ts["actions_taken"].append("skip")
            reward_breakdown["skip"] = self.PENALTY_SKIP
            messages.append("Ticket skipped — penalty applied")
            self._advance_ticket()

        # ── Compute total step reward ──────────────────────────────────────────
        step_reward = sum(reward_breakdown.values())
        self.total_reward += step_reward

        # ── Check for episode completion (step limit or all done) ──────────────
        all_processed = self.current_ticket is None and len(self.ticket_queue) == 0

        if all_processed:
            reward_breakdown["completion_bonus"] = self.BONUS_ALL_COMPLETE
            step_reward += self.BONUS_ALL_COMPLETE
            self.total_reward += self.BONUS_ALL_COMPLETE
            messages.append("All tickets processed — completion bonus!")
            self.done = True

        if self.step_count >= self.task["max_steps"]:
            self.done = True
            if not all_processed:
                messages.append(f"Step limit reached ({self.task['max_steps']} steps)")

        reward = Reward(
            total=round(step_reward, 4),
            breakdown=reward_breakdown,
            message=" | ".join(messages),
        )

        # Record trajectory
        self.trajectory.append({
            "step": self.step_count,
            "action": action.model_dump(),
            "reward": reward.model_dump(),
        })

        obs = self._make_observation()
        return obs, reward, self.done, {"step": self.step_count, "message": " | ".join(messages)}

    # ── state ──────────────────────────────────────────────────────────────────

    def state(self) -> StateResponse:
        if not self._initialized:
            raise RuntimeError("Call reset() first")
        return StateResponse(
            session_id="",  # filled by the API layer
            task_id=self.task_id,
            step=self.step_count,
            max_steps=self.task["max_steps"],
            ticket_queue=list(self.ticket_queue),
            current_ticket=self.current_ticket,
            ticket_states=copy.deepcopy(self.ticket_states),
            total_reward=round(self.total_reward, 4),
            done=self.done,
            trajectory=self.trajectory,
        )

    # ── grade ──────────────────────────────────────────────────────────────────

    def grade(self) -> Dict:
        """Run the task's grader on the current ticket_states."""
        return grader_module.grade(self.task_id, self.ticket_states)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _advance_ticket(self):
        """Move to the next ticket in the queue, or set done if queue empty."""
        if self.ticket_queue:
            self.current_ticket = self.ticket_queue.pop(0)
        else:
            self.current_ticket = None

    def _make_observation(self) -> Observation:
        """Build the Observation object from current state."""
        available_actions = self._available_actions()
        return Observation(
            task_id=self.task_id,
            task_description=self.task["description"],
            step=self.step_count,
            max_steps=self.task["max_steps"],
            current_ticket=self.current_ticket,
            queue_size=len(self.ticket_queue),
            processed_count=sum(
                1 for ts in self.ticket_states.values() if ts.get("resolved")
                or (self.current_ticket and ts["ticket_id"] != self.current_ticket.id
                    and not ts["resolved"] and ts["actions_taken"])
            ),
            ticket_states=copy.deepcopy(self.ticket_states),
            available_actions=available_actions,
            done=self.done,
            info={
                "objective": self.task["objective"],
                "total_reward_so_far": round(self.total_reward, 4),
                "reward_weights": self.task["reward_weights"],
            },
        )

    def _available_actions(self) -> List[str]:
        """Return sensible list of valid next actions given current state."""
        if self.done or self.current_ticket is None:
            return []
        tid = self.current_ticket.id
        ts = self.ticket_states.get(tid, {})
        taken = set(ts.get("actions_taken", []))
        required = set(self.task.get("required_actions", []))
        optional = set(self.task.get("optional_actions", []))
        all_possible = required | optional | {"skip"}
        return sorted(a for a in all_possible if a not in taken or a == "resolve")

    def _invalid(self, message: str) -> Tuple[Observation, Reward, bool, Dict]:
        r = Reward(
            total=self.PENALTY_INVALID_ACTION,
            breakdown={"invalid_action": self.PENALTY_INVALID_ACTION},
            message=message,
        )
        self.total_reward += r.total
        return self._make_observation(), r, self.done, {"error": message}
