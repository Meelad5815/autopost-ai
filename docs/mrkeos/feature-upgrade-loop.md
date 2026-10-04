# MRK Feature Upgrade Loop

1. Read ranked feature-gap signals.
2. Select one candidate.
3. Produce an implementation plan and acceptance criteria.
4. Generate an isolated upgrade candidate; never overwrite production automatically.
5. Run smoke, SEO and lightweight performance checks.
6. Store the candidate in `data/upgrade_candidates.json`.
7. Require explicit human approval.
8. Deployment is a separate, gated operation.

The loop is intentionally free-first and conservative. Public feature requests are
signals, not permission to copy proprietary code or automatically deploy changes.

## Safety gate

The upgrade engine must not introduce password collection, OTP bypass, payment
transfer, identity-verification bypass, or other sensitive automation.

## Suggested future stages

- Build candidate changes in a dedicated branch.
- Open a draft pull request with test results.
- After human approval, merge/deploy.
- Keep rollback metadata and an upgrade history.
