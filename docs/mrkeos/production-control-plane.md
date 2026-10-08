# MRK Production Upgrade Control Plane

The upgrade system is split into isolated candidate generation, human review, approved production change, deployment, and rollback.

## Current production safety model

- Candidate generation never changes production.
- Candidate PRs are created as drafts.
- A PR must be manually reviewed and marked ready before any production-impacting action.
- The existing deployment gate requires an explicit workflow dispatch approval.
- Upgrade history is stored in data/upgrade_history.json.
- Rollback uses a normal Git revert workflow/PR rather than rewriting Git history.
- No password, OTP, bank, payment-transfer, or identity-verification automation is permitted.

## Audit record

Each completed production change should record:

- candidate_id
- PR number and URL
- source branch and target branch
- previous known-good commit
- resulting commit
- deployment target and status
- rollback reference
- timestamps

## Operating rule

Do not merge or deploy a candidate merely because automation generated it. Human review remains the final authorization boundary.

GitHub Actions schedules are best-effort. A 15-minute schedule is a target cadence, not a guarantee of exact execution.
