# MRK Website Manager

This module is the autonomous website operations layer for MRK Autopost AI.

## Responsibilities

1. Verify the live Laravel website.
2. Check the main public routes.
3. Detect common Laravel/PHP error signatures in responses.
4. Inspect the latest deployment workflow status of the Laravel repository.
5. Create a GitHub issue when production health or deployment verification fails.
6. Run automatically every 5 minutes and after an explicit repository dispatch.

## Safety model

The manager is diagnostic-first. It does not blindly rewrite production code.

The intended lifecycle is:

Detect -> Diagnose -> Apply only a known safe remediation -> Test -> Deploy -> Verify -> Record

Cross-repository code changes and deployment dispatches require an explicit GitHub authorization mechanism (for example a GitHub App or a least-privilege token). That authorization must never be stored in source code.

## Important limit

GitHub Actions scheduled workflows have a minimum supported schedule interval of 5 minutes. Therefore this cannot literally execute every second; event-driven checks can run sooner when a deployment or repository event triggers them.

## Current monitored Laravel project

- Repository: Meelad5815/MRK-Laravel-Portal
- Production host: meeladrazakhokhar786.freehosting.dev

## Future safe remediation layers

- Retry transient deployment failures.
- Detect missing Laravel storage directories.
- Detect missing APP_KEY.
- Detect invalid production environment defaults.
- Detect failed asset builds.
- Run Laravel route/config checks before deployment.
- Verify production immediately after deployment.
- Only then consider AI-assisted code repair behind tests and a rollback path.
