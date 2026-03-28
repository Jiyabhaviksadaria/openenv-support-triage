"""
Ticket pool with ground truth labels.
Ground truth is used by graders — never exposed to the agent directly.
"""
from typing import List, Dict

TICKET_POOL: List[Dict] = [
    # ── BILLING TICKETS ────────────────────────────────────────────────────────
    {
        "id": "T001",
        "subject": "Charged twice for my subscription this month",
        "body": (
            "Hello, I noticed two charges of $29.99 on my credit card statement "
            "dated January 3rd and January 5th. I only have one active subscription "
            "and have been a customer since 2022. Please issue a refund for the "
            "duplicate charge as soon as possible. My account email is jsmith@gmail.com."
        ),
        "sender": "jsmith@gmail.com",
        "timestamp": "2025-01-06T08:14:00Z",
        "attachments": ["bank_statement.pdf"],
        "ground_truth": {
            "category": "billing",
            "priority": "high",
            "department": "billing_team",
            "requires_escalation": False,
            "response_keywords": ["refund", "charge", "apologize", "investigate", "billing"],
            "difficulty": "easy",
        },
    },
    {
        "id": "T002",
        "subject": "Question about enterprise pricing",
        "body": (
            "Hi team, we're evaluating your product for our 500-person engineering org. "
            "Could you share information about enterprise pricing tiers and volume discounts? "
            "We're particularly interested in SSO/SAML support and dedicated account management. "
            "Happy to schedule a call."
        ),
        "sender": "procurement@bigcorp.com",
        "timestamp": "2025-01-06T09:30:00Z",
        "attachments": [],
        "ground_truth": {
            "category": "billing",
            "priority": "medium",
            "department": "billing_team",
            "requires_escalation": False,
            "response_keywords": ["enterprise", "pricing", "account", "schedule", "team"],
            "difficulty": "easy",
        },
    },
    {
        "id": "T003",
        "subject": "Payment method declined — cannot access account",
        "body": (
            "My credit card expired and I updated it but the system keeps saying payment "
            "declined and now I can't access any of my projects. I have an important deadline "
            "tomorrow morning. I've tried three different cards. This is urgent!!"
        ),
        "sender": "maya.chen@startup.io",
        "timestamp": "2025-01-06T17:55:00Z",
        "attachments": [],
        "ground_truth": {
            "category": "billing",
            "priority": "urgent",
            "department": "billing_team",
            "requires_escalation": True,
            "response_keywords": ["payment", "access", "immediately", "priority", "urgent", "resolve"],
            "difficulty": "medium",
        },
    },

    # ── TECHNICAL TICKETS ──────────────────────────────────────────────────────
    {
        "id": "T004",
        "subject": "API rate limits — feature request for higher limits",
        "body": (
            "Love the product! One thing that would help us a lot is higher rate limits "
            "on the REST API. We're currently capped at 100 req/min which is becoming a "
            "bottleneck for our pipeline. Would you consider a higher tier or burst option? "
            "Not urgent, just flagging for the roadmap."
        ),
        "sender": "eng@techstartup.com",
        "timestamp": "2025-01-06T11:00:00Z",
        "attachments": [],
        "ground_truth": {
            "category": "technical",
            "priority": "low",
            "department": "engineering",
            "requires_escalation": False,
            "response_keywords": ["rate limit", "roadmap", "feedback", "engineering", "consider"],
            "difficulty": "easy",
        },
    },
    {
        "id": "T005",
        "subject": "Export to CSV producing malformed files",
        "body": (
            "When I export reports to CSV, cells containing commas are not being properly "
            "quoted, so the file breaks in Excel. Steps to reproduce: 1) Go to Reports, "
            "2) Filter by any date range, 3) Click Export CSV, 4) Open in Excel — columns "
            "are misaligned. Running Chrome 120 on macOS 14.2. Attaching example file."
        ),
        "sender": "reports_user@company.org",
        "timestamp": "2025-01-06T10:15:00Z",
        "attachments": ["malformed_export.csv"],
        "ground_truth": {
            "category": "technical",
            "priority": "medium",
            "department": "engineering",
            "requires_escalation": False,
            "response_keywords": ["bug", "reproduce", "fix", "engineering", "workaround", "csv"],
            "difficulty": "easy",
        },
    },
    {
        "id": "T006",
        "subject": "PRODUCTION DOWN — all API calls returning 503",
        "body": (
            "CRITICAL: Our entire production system is down. Every API call returns 503 "
            "Service Unavailable since approximately 14:32 UTC. This is affecting 50,000+ "
            "end users. We are losing $10,000 per minute. Status page shows all green "
            "which is clearly wrong. Need immediate response from engineering. "
            "Contact: CTO direct line +1-555-0100."
        ),
        "sender": "oncall@enterprise-client.com",
        "timestamp": "2025-01-06T14:45:00Z",
        "attachments": ["error_logs.txt"],
        "ground_truth": {
            "category": "technical",
            "priority": "urgent",
            "department": "engineering",
            "requires_escalation": True,
            "response_keywords": [
                "immediately", "escalate", "engineering", "status", "investigating", "priority", "contact"
            ],
            "difficulty": "hard",
        },
    },
    {
        "id": "T007",
        "subject": "Dark mode flickers on Safari",
        "body": (
            "Minor cosmetic issue — when switching between tabs in Safari 17 with dark mode "
            "enabled, there's a brief white flash. Doesn't affect functionality at all, "
            "just a bit annoying. MacBook Pro M3, macOS Sonoma."
        ),
        "sender": "casual_user@email.com",
        "timestamp": "2025-01-06T12:30:00Z",
        "attachments": [],
        "ground_truth": {
            "category": "technical",
            "priority": "low",
            "department": "engineering",
            "requires_escalation": False,
            "response_keywords": ["noted", "bug", "team", "fix", "cosmetic"],
            "difficulty": "easy",
        },
    },

    # ── ACCOUNT TICKETS ────────────────────────────────────────────────────────
    {
        "id": "T008",
        "subject": "How do I add team members to my workspace?",
        "body": (
            "Hi! I just upgraded to the Team plan and want to invite my colleagues. "
            "I went to Settings > Team but the 'Invite' button is grayed out. "
            "Am I missing something? Thanks!"
        ),
        "sender": "newuser@company.com",
        "timestamp": "2025-01-06T09:00:00Z",
        "attachments": [],
        "ground_truth": {
            "category": "account",
            "priority": "low",
            "department": "general_support",
            "requires_escalation": False,
            "response_keywords": ["invite", "team", "settings", "admin", "steps"],
            "difficulty": "easy",
        },
    },
    {
        "id": "T009",
        "subject": "Account locked after multiple login attempts",
        "body": (
            "I forgot my password and tried too many times. Now my account says it's "
            "locked for 24 hours but I need access NOW for a client presentation in 2 hours. "
            "Can you unlock it manually? Account email: locked.user@agency.com"
        ),
        "sender": "locked.user@agency.com",
        "timestamp": "2025-01-06T13:20:00Z",
        "attachments": [],
        "ground_truth": {
            "category": "account",
            "priority": "high",
            "department": "account_management",
            "requires_escalation": False,
            "response_keywords": ["unlock", "account", "identity", "verify", "access", "reset"],
            "difficulty": "medium",
        },
    },
    {
        "id": "T010",
        "subject": "Suspicious login from unknown location — possible breach",
        "body": (
            "I just received a login notification from Singapore (I'm in Boston). "
            "I did NOT authorize this. I've already changed my password but I'm worried "
            "my data may have been accessed. I have sensitive client contracts stored here. "
            "Please investigate immediately and let me know if data was accessed."
        ),
        "sender": "worried.user@lawfirm.com",
        "timestamp": "2025-01-06T15:10:00Z",
        "attachments": ["login_notification.png"],
        "ground_truth": {
            "category": "security",
            "priority": "urgent",
            "department": "security_team",
            "requires_escalation": True,
            "response_keywords": [
                "security", "investigate", "sessions", "immediately", "team", "access logs", "protect"
            ],
            "difficulty": "hard",
        },
    },

    # ── GENERAL TICKETS ────────────────────────────────────────────────────────
    {
        "id": "T011",
        "subject": "Love the product — just wanted to say thanks!",
        "body": (
            "Not a support request, just wanted to pass along that your product has saved "
            "our team countless hours. The recent v3.0 update was especially great. "
            "Keep up the great work!"
        ),
        "sender": "happy.customer@smallbiz.com",
        "timestamp": "2025-01-06T10:45:00Z",
        "attachments": [],
        "ground_truth": {
            "category": "general",
            "priority": "low",
            "department": "general_support",
            "requires_escalation": False,
            "response_keywords": ["thank", "appreciate", "feedback", "team"],
            "difficulty": "easy",
        },
    },
    {
        "id": "T012",
        "subject": "Request: Bulk delete feature for old projects",
        "body": (
            "Would love a way to select multiple old projects and archive/delete them "
            "in bulk. Right now I have to do them one-by-one which takes forever. "
            "Happy to beta test if you build this!"
        ),
        "sender": "power.user@freelance.com",
        "timestamp": "2025-01-06T11:30:00Z",
        "attachments": [],
        "ground_truth": {
            "category": "general",
            "priority": "low",
            "department": "general_support",
            "requires_escalation": False,
            "response_keywords": ["feature request", "roadmap", "noted", "team", "feedback"],
            "difficulty": "easy",
        },
    },

    # ── COMPLEX / AMBIGUOUS TICKETS (for hard task) ────────────────────────────
    {
        "id": "T013",
        "subject": "Refund request + account deletion after data concern",
        "body": (
            "After reading about your recent data incident in the news, I want to "
            "immediately cancel my subscription, get a full refund for this month ($49.99), "
            "AND have all my data permanently deleted per GDPR Article 17. "
            "I expect written confirmation within 24 hours. If I don't hear back, "
            "I'll be filing a complaint with the ICO."
        ),
        "sender": "gdpr.concerned@eu-user.de",
        "timestamp": "2025-01-06T16:00:00Z",
        "attachments": [],
        "ground_truth": {
            "category": "security",
            "priority": "urgent",
            "department": "security_team",
            "requires_escalation": True,
            "response_keywords": [
                "GDPR", "data deletion", "refund", "confirm", "legal", "24 hours", "privacy"
            ],
            "difficulty": "hard",
        },
    },
    {
        "id": "T014",
        "subject": "API integration broken after your update — affecting paying customers",
        "body": (
            "Your v2.8 API update yesterday broke our integration. "
            "The 'batch_process' endpoint now returns a 400 error when we pass more than "
            "50 items (it used to allow 500). This is a breaking change with no notice! "
            "We have an SLA with our own clients. I need: 1) Immediate rollback or hotfix, "
            "2) Written acknowledgment of the breaking change, 3) Compensation for downtime. "
            "We're on the Enterprise plan paying $2,000/month."
        ),
        "sender": "cto@b2b-customer.com",
        "timestamp": "2025-01-06T14:00:00Z",
        "attachments": ["api_error_logs.txt", "sla_agreement.pdf"],
        "ground_truth": {
            "category": "technical",
            "priority": "urgent",
            "department": "engineering",
            "requires_escalation": True,
            "response_keywords": [
                "breaking change", "investigate", "engineering", "SLA", "priority",
                "hotfix", "compensat", "apologize"
            ],
            "difficulty": "hard",
        },
    },
    {
        "id": "T015",
        "subject": "Cannot export data — trial expired but I haven't decided yet",
        "body": (
            "My 14-day trial just ended and now I can't export any of my data. "
            "I understand I need to pay to continue using features, but I should be "
            "able to export my own data regardless! I have 3 months of work in here. "
            "Please restore read-only export access immediately."
        ),
        "sender": "trial.user@personal.com",
        "timestamp": "2025-01-06T12:00:00Z",
        "attachments": [],
        "ground_truth": {
            "category": "account",
            "priority": "high",
            "department": "account_management",
            "requires_escalation": False,
            "response_keywords": ["export", "data", "access", "trial", "help", "resolve"],
            "difficulty": "medium",
        },
    },
]

# Convenience lookup
TICKET_BY_ID: Dict[str, Dict] = {t["id"]: t for t in TICKET_POOL}

# Task-specific ticket assignments
TASK_TICKETS = {
    "single_triage": ["T005"],          # Easy: one clear billing bug report
    "queue_triage": ["T001", "T004", "T007", "T009", "T011"],  # Medium: mixed queue
    "full_resolution": ["T006", "T010", "T014"],                # Hard: complex, high-stakes
}
