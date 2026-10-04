# MRK AI Earning OS

MRK AI Earning OS is the earning/automation layer of AutoPost AI.

## Design goals
- Run primarily on GitHub Actions so the laptop can be offline.
- Prefer free/open-source services and deterministic automation.
- Never store passwords, OTPs, bank details, payment credentials, or identity documents in the repository.
- Separate autonomous tasks from human-approval tasks.
- Produce auditable JSON/Markdown outputs instead of silently performing risky account actions.

## Modules
1. Opportunity Hunter — scans public feeds, ranks opportunities, and writes reports.
2. Website Factory — prepares site specifications, content plans, SEO plans and deployment manifests.
3. Content Engine — research -> topic selection -> article -> SEO -> media -> publishing queue.
4. Freelance Engine — prepares service packages, proposals, portfolio pages and lead lists.
5. Social/Video Engine — converts approved research into platform-specific content.
6. Control Plane — tracks tasks, approvals, errors and outcomes.

## Security boundary
The agent must not request, infer, expose or store banking credentials; bypass OTP/CAPTCHA/identity verification; automate prohibited platform behavior; or make financial transfers without explicit user action.

## First milestone
The first milestone is a cloud-running Opportunity Hunter. It needs no paid AI API and no always-on laptop.
