"""
Test suite for the Customer Support Triage OpenEnv.
Run: python test_environment.py
All tests are deterministic and require no external API calls.
"""
import sys
import json

# ─── Helpers ──────────────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
results = []


def test(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except AssertionError as e:
        print(f"  {FAIL} {name} — ASSERTION: {e}")
        results.append((name, False, str(e)))
    except Exception as e:
        print(f"  {FAIL} {name} — ERROR: {e}")
        results.append((name, False, str(e)))


# ─── Tests ────────────────────────────────────────────────────────────────────

def run_tests():

    print("\n" + "="*65)
    print("CUSTOMER SUPPORT TRIAGE — OPENENV TEST SUITE")
    print("="*65)

    # ── Section 1: Ticket Data ──────────────────────────────────────────────────
    print("\n[1] Ticket Data Integrity")
    
    from tickets_data import TICKET_POOL, TICKET_BY_ID, TASK_TICKETS

    def test_ticket_count():
        assert len(TICKET_POOL) >= 15, f"Expected ≥15 tickets, got {len(TICKET_POOL)}"

    def test_ticket_ids_unique():
        ids = [t["id"] for t in TICKET_POOL]
        assert len(ids) == len(set(ids)), "Duplicate ticket IDs found"

    def test_ground_truth_fields():
        required = {"category", "priority", "department", "requires_escalation", "response_keywords"}
        for t in TICKET_POOL:
            missing = required - set(t["ground_truth"].keys())
            assert not missing, f"Ticket {t['id']} missing ground truth fields: {missing}"

    def test_valid_categories():
        valid = {"billing", "technical", "account", "general", "security"}
        for t in TICKET_POOL:
            cat = t["ground_truth"]["category"]
            assert cat in valid, f"Ticket {t['id']} has invalid category: {cat!r}"

    def test_valid_priorities():
        valid = {"low", "medium", "high", "urgent"}
        for t in TICKET_POOL:
            pri = t["ground_truth"]["priority"]
            assert pri in valid, f"Ticket {t['id']} has invalid priority: {pri!r}"

    def test_valid_departments():
        valid = {"billing_team", "engineering", "account_management", "security_team", "general_support"}
        for t in TICKET_POOL:
            dep = t["ground_truth"]["department"]
            assert dep in valid, f"Ticket {t['id']} has invalid department: {dep!r}"

    def test_task_ticket_references():
        for task, tids in TASK_TICKETS.items():
            for tid in tids:
                assert tid in TICKET_BY_ID, f"Task {task} references unknown ticket {tid}"

    def test_three_tasks_defined():
        assert len(TASK_TICKETS) == 3, f"Expected 3 tasks, got {len(TASK_TICKETS)}"

    def test_response_keywords_nonempty():
        for t in TICKET_POOL:
            kws = t["ground_truth"].get("response_keywords", [])
            assert len(kws) >= 2, f"Ticket {t['id']} has too few response keywords: {kws}"

    test("15+ tickets in pool", test_ticket_count)
    test("Ticket IDs are unique", test_ticket_ids_unique)
    test("All tickets have required ground truth fields", test_ground_truth_fields)
    test("All categories are valid enum values", test_valid_categories)
    test("All priorities are valid enum values", test_valid_priorities)
    test("All departments are valid enum values", test_valid_departments)
    test("All task ticket references exist", test_task_ticket_references)
    test("Exactly 3 tasks defined", test_three_tasks_defined)
    test("All tickets have ≥2 response keywords", test_response_keywords_nonempty)

    # ── Section 2: Grader Correctness ──────────────────────────────────────────
    print("\n[2] Grader Logic")
    
    from graders import grade, grade_response
    from tickets_data import TICKET_BY_ID

    def test_single_perfect():
        states = {"T005": {"category": "technical", "priority": "medium",
                           "department": None, "response_text": None,
                           "escalated": False, "resolved": False, "actions_taken": []}}
        r = grade("single_triage", states)
        assert r["score"] == 1.0, f"Expected 1.0, got {r['score']}"

    def test_single_half():
        states = {"T005": {"category": "billing", "priority": "medium",
                           "department": None, "response_text": None,
                           "escalated": False, "resolved": False, "actions_taken": []}}
        r = grade("single_triage", states)
        assert r["score"] == 0.5, f"Expected 0.5, got {r['score']}"

    def test_single_zero():
        states = {"T005": {"category": "billing", "priority": "high",
                           "department": None, "response_text": None,
                           "escalated": False, "resolved": False, "actions_taken": []}}
        r = grade("single_triage", states)
        assert r["score"] == 0.0, f"Expected 0.0, got {r['score']}"

    def test_single_empty():
        r = grade("single_triage", {})
        assert r["score"] == 0.0, f"Expected 0.0, got {r['score']}"

    def test_queue_perfect():
        perfect = {
            "T001": {"category": "billing",   "priority": "high",   "department": "billing_team"},
            "T004": {"category": "technical", "priority": "low",    "department": "engineering"},
            "T007": {"category": "technical", "priority": "low",    "department": "engineering"},
            "T009": {"category": "account",   "priority": "high",   "department": "account_management"},
            "T011": {"category": "general",   "priority": "low",    "department": "general_support"},
        }
        for tid in perfect:
            perfect[tid].update({"response_text": None, "escalated": False, "resolved": False, "actions_taken": []})
        r = grade("queue_triage", perfect)
        assert abs(r["score"] - 1.0) < 0.01, f"Expected ~1.0, got {r['score']}"

    def test_queue_score_in_range():
        states = {tid: {"category": "billing", "priority": "urgent", "department": "engineering",
                        "response_text": None, "escalated": False, "resolved": False, "actions_taken": []}
                  for tid in ["T001","T004","T007","T009","T011"]}
        r = grade("queue_triage", states)
        assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    def test_full_resolution_score_range():
        states = {}
        for tid in ["T006","T010","T014"]:
            states[tid] = {
                "category": TICKET_BY_ID[tid]["ground_truth"]["category"],
                "priority": "urgent",
                "department": TICKET_BY_ID[tid]["ground_truth"]["department"],
                "response_text": "We sincerely apologize and our engineering team is investigating immediately as top priority. We will update you within 30 minutes and escalate to all relevant teams.",
                "escalated": True,
                "resolved": True,
                "actions_taken": ["classify","set_priority","route","respond","escalate","resolve"],
            }
        r = grade("full_resolution", states)
        assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"
        assert r["score"] > 0.5, f"Good agent should score > 0.5, got {r['score']}"

    def test_grader_deterministic():
        states = {"T005": {"category": "technical", "priority": "medium",
                           "department": None, "response_text": None,
                           "escalated": False, "resolved": False, "actions_taken": []}}
        r1 = grade("single_triage", states)
        r2 = grade("single_triage", states)
        assert r1["score"] == r2["score"], "Grader is not deterministic"

    def test_unknown_task_raises():
        try:
            grade("nonexistent_task", {})
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    test("single_triage perfect score = 1.0", test_single_perfect)
    test("single_triage half score = 0.5", test_single_half)
    test("single_triage zero score = 0.0", test_single_zero)
    test("single_triage empty states = 0.0", test_single_empty)
    test("queue_triage perfect score ≈ 1.0", test_queue_perfect)
    test("queue_triage score always in [0, 1]", test_queue_score_in_range)
    test("full_resolution good agent scores > 0.5", test_full_resolution_score_range)
    test("Grader is deterministic", test_grader_deterministic)
    test("Unknown task raises ValueError", test_unknown_task_raises)

    # ── Section 3: Response Quality Grader ─────────────────────────────────────
    print("\n[3] Response Quality Grader")

    def test_empty_response_zero():
        score, _ = grade_response("", TICKET_BY_ID["T006"])
        assert score == 0.0

    def test_short_response_zero():
        score, _ = grade_response("ok", TICKET_BY_ID["T006"])
        assert score == 0.0

    def test_good_response_high_score():
        resp = ("We sincerely apologize for this production outage affecting your users. "
                "Our engineering team is investigating the 503 errors as top priority immediately. "
                "We will contact your team with a status update within 30 minutes and escalate "
                "to our senior engineers right now. This is our highest priority incident.")
        score, _ = grade_response(resp, TICKET_BY_ID["T006"])
        assert score >= 0.6, f"Good response should score ≥ 0.6, got {score}"

    def test_poor_response_low_score():
        resp = "Hello there. Thank you for your message. We have received it."
        score, _ = grade_response(resp, TICKET_BY_ID["T006"])
        assert score <= 0.4, f"Poor response should score ≤ 0.4, got {score}"

    def test_response_score_in_range():
        for t in TICKET_POOL:
            score, _ = grade_response("Test response", t)
            assert 0.0 <= score <= 1.0, f"Response score out of range for {t['id']}: {score}"

    test("Empty response scores 0.0", test_empty_response_zero)
    test("Too-short response scores 0.0", test_short_response_zero)
    test("Good response scores ≥ 0.6", test_good_response_high_score)
    test("Poor response scores ≤ 0.4", test_poor_response_low_score)
    test("All response scores in [0, 1]", test_response_score_in_range)

    # ── Section 4: Task Definitions ─────────────────────────────────────────────
    print("\n[4] Task Definitions")

    from tasks import TASKS, TASKS_BY_ID

    def test_three_tasks():
        assert len(TASKS) == 3, f"Expected 3 tasks, got {len(TASKS)}"

    def test_difficulty_progression():
        difficulties = [t["difficulty"] for t in TASKS]
        expected = ["easy", "medium", "hard"]
        assert difficulties == expected, f"Expected {expected}, got {difficulties}"

    def test_step_limits_increasing():
        steps = [t["max_steps"] for t in TASKS]
        assert steps == sorted(steps), f"Step limits should increase: {steps}"

    def test_required_fields():
        for t in TASKS:
            for field in ["task_id", "name", "difficulty", "description", "objective",
                          "ticket_ids", "max_steps", "required_actions", "reward_weights", "action_schema"]:
                assert field in t, f"Task {t.get('task_id')} missing field: {field}"

    def test_reward_weights_sum():
        for t in TASKS:
            total = sum(t["reward_weights"].values())
            assert abs(total - 1.0) < 0.01, f"Task {t['task_id']} weights sum to {total}, not 1.0"

    def test_action_schema_complete():
        for t in TASKS:
            schema = t["action_schema"]
            assert "action_type" in schema
            assert "ticket_id" in schema
            assert "enum" in schema["action_type"]

    test("Exactly 3 tasks", test_three_tasks)
    test("Difficulty progression: easy → medium → hard", test_difficulty_progression)
    test("Step limits increase by difficulty", test_step_limits_increasing)
    test("All tasks have required metadata fields", test_required_fields)
    test("Reward weights sum to 1.0 per task", test_reward_weights_sum)
    test("Action schema is complete", test_action_schema_complete)

    # ── Section 5: OpenEnv YAML ─────────────────────────────────────────────────
    print("\n[5] openenv.yaml Spec Compliance")

    import yaml

    def test_yaml_parseable():
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        return data

    def test_yaml_required_fields():
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        for field in ["name", "version", "description", "tasks", "observation_space",
                      "action_space", "reward_space", "endpoints"]:
            assert field in data, f"openenv.yaml missing field: {field}"

    def test_yaml_three_tasks():
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        assert len(data["tasks"]) == 3

    def test_yaml_endpoints():
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        eps = data["endpoints"]
        for ep in ["reset", "step", "state", "tasks", "grader", "baseline"]:
            assert ep in eps, f"Missing endpoint in yaml: {ep}"

    test("YAML is parseable", test_yaml_parseable)
    test("YAML has all required top-level fields", test_yaml_required_fields)
    test("YAML defines 3 tasks", test_yaml_three_tasks)
    test("YAML specifies all required endpoints", test_yaml_endpoints)

    # ── Section 6: Dockerfile ───────────────────────────────────────────────────
    print("\n[6] Dockerfile")

    def test_dockerfile_exists():
        with open("Dockerfile") as f:
            content = f.read()
        assert len(content) > 100

    def test_dockerfile_port():
        with open("Dockerfile") as f:
            content = f.read()
        assert "7860" in content, "Dockerfile must expose port 7860 (HF Spaces requirement)"

    def test_dockerfile_has_cmd():
        with open("Dockerfile") as f:
            content = f.read()
        assert "CMD" in content or "ENTRYPOINT" in content

    def test_dockerfile_has_healthcheck():
        with open("Dockerfile") as f:
            content = f.read()
        assert "HEALTHCHECK" in content

    def test_requirements_has_fastapi():
        with open("requirements.txt") as f:
            reqs = f.read()
        assert "fastapi" in reqs.lower()
        assert "uvicorn" in reqs.lower()
        assert "pydantic" in reqs.lower()

    test("Dockerfile exists and has content", test_dockerfile_exists)
    test("Dockerfile exposes port 7860", test_dockerfile_port)
    test("Dockerfile has CMD or ENTRYPOINT", test_dockerfile_has_cmd)
    test("Dockerfile has HEALTHCHECK", test_dockerfile_has_healthcheck)
    test("requirements.txt has fastapi + uvicorn + pydantic", test_requirements_has_fastapi)

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("="*65)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)
    print(f"RESULTS: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED:")
        for name, ok, err in results:
            if not ok:
                print(f"    ✗ {name}: {err}")
    else:
        print("  — ALL TESTS PASSED ✓")
    print("="*65)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
