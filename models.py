"""
Typed Pydantic models for the Customer Support Triage OpenEnv environment.
Implements the full OpenEnv spec: Action, Observation, Reward.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Enumerations ─────────────────────────────────────────────────────────────

class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    general = "general"
    security = "security"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class Department(str, Enum):
    billing_team = "billing_team"
    engineering = "engineering"
    account_management = "account_management"
    security_team = "security_team"
    general_support = "general_support"


class ActionType(str, Enum):
    classify = "classify"            # Set ticket category
    set_priority = "set_priority"    # Set urgency level
    route = "route"                  # Route to department
    respond = "respond"              # Draft customer response
    escalate = "escalate"            # Escalate to senior staff
    resolve = "resolve"              # Mark ticket as resolved
    skip = "skip"                    # Skip current ticket (penalty)


# ─── Core OpenEnv Models ───────────────────────────────────────────────────────

class Action(BaseModel):
    """
    Agent action. Required fields vary by action_type:
      - classify:      ticket_id + category
      - set_priority:  ticket_id + priority
      - route:         ticket_id + department
      - respond:       ticket_id + response_text
      - escalate:      ticket_id
      - resolve:       ticket_id + resolution_notes (optional)
      - skip:          ticket_id
    """
    action_type: ActionType = Field(..., description="Type of action to perform")
    ticket_id: str = Field(..., description="ID of the ticket to act on")
    category: Optional[Category] = Field(None, description="Category for 'classify' action")
    priority: Optional[Priority] = Field(None, description="Priority for 'set_priority' action")
    department: Optional[Department] = Field(None, description="Department for 'route' action")
    response_text: Optional[str] = Field(None, description="Customer response text for 'respond' action")
    resolution_notes: Optional[str] = Field(None, description="Notes for 'resolve' action")


class Ticket(BaseModel):
    """A customer support ticket presented to the agent."""
    id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    attachments: List[str] = Field(default_factory=list)


class TicketState(BaseModel):
    """Internal tracking of what the agent has done to a ticket."""
    ticket_id: str
    category: Optional[str] = None
    priority: Optional[str] = None
    department: Optional[str] = None
    response_text: Optional[str] = None
    escalated: bool = False
    resolved: bool = False
    actions_taken: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    """
    Observation returned after reset() or step().
    Contains the current ticket, queue status, and available actions.
    """
    task_id: str = Field(..., description="Current task identifier")
    task_description: str = Field(..., description="Human-readable task objective")
    step: int = Field(..., description="Current step number within episode")
    max_steps: int = Field(..., description="Maximum steps allowed")
    current_ticket: Optional[Ticket] = Field(None, description="Active ticket the agent should process")
    queue_size: int = Field(0, description="Number of tickets remaining in queue")
    processed_count: int = Field(0, description="Number of tickets fully processed")
    ticket_states: Dict[str, Dict] = Field(default_factory=dict, description="Current state of each ticket")
    available_actions: List[str] = Field(default_factory=list, description="Valid action_types for current state")
    done: bool = Field(False, description="Whether episode is complete")
    info: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    """Reward signal with breakdown for interpretability."""
    total: float = Field(..., ge=-1.0, le=1.0, description="Total reward for this step")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Per-component reward breakdown")
    message: str = Field("", description="Human-readable explanation")


class StepResponse(BaseModel):
    """Full response from step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    """Response from reset()."""
    session_id: str
    observation: Observation
    task_id: str
    task_description: str


class StateResponse(BaseModel):
    """Full internal state (for debugging / state() endpoint)."""
    session_id: str
    task_id: str
    step: int
    max_steps: int
    ticket_queue: List[Ticket]
    current_ticket: Optional[Ticket]
    ticket_states: Dict[str, Dict]
    total_reward: float
    done: bool
    trajectory: List[Dict]


class GraderResponse(BaseModel):
    """Score returned by the grader endpoint after episode completion."""
    session_id: str
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, Any]
    feedback: str


class TaskInfo(BaseModel):
    """Task metadata returned by /tasks endpoint."""
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    action_schema: Dict[str, Any]
    objective: str
