# MRK Autopost AI

Zero-cost remote WordPress autoposting system designed to run from **GitHub Actions**, so the user's personal computer does not need to stay powered on.

## How it runs

GitHub Actions starts a short-lived Ubuntu runner at the configured UTC times:

- 10:00 PKT
- 12:00 PKT
- 14:00 PKT
- 16:00 PKT
- 18:00 PKT

Each slot runs the scheduler once. The scheduler creates one post in each configured language, then performs the update loop. The personal Windows PC is not required for scheduled production runs.

## AI modes

### Default: local zero-cost engine

`AI_PROVIDER=local`

This mode requires no OpenAI, Gemini, or other paid AI API. It uses the repository's deterministic local content-intelligence engine and can run inside GitHub Actions.

Important: this is not a neural LLM equivalent to ChatGPT. It is a deterministic generation system designed for zero-cost automation.

### Optional: self-hosted Ollama

`AI_PROVIDER=ollama`

Ollama can provide a real local open-source LLM on a computer/server that stays online. A personal PC that is switched off cannot provide Ollama inference to GitHub Actions.

## Required GitHub Secrets

Add these repository secrets:

- `WP_URL`
- `WP_USER`
- `WP_APP_PASSWORD`

No paid AI API key is required for the GitHub Actions workflow.

## Publishing

The GitHub production workflow explicitly uses:

`POST_STATUS=publish`

The repository configuration remains conservative by default for local/manual testing.

## Important operational notes

- GitHub Actions is a scheduled job runner, not a permanently running process.
- Scheduled runs can occasionally be delayed by GitHub.
- WordPress must be reachable from the internet.
- WordPress Application Password credentials must remain in GitHub Secrets, never in source code.
- The workflow stores runtime history/trends/keywords back into the repository.
- The current traffic metrics are not fabricated; real analytics integration is still a separate feature.

## Recommended production path

1. Keep GitHub Actions as the remote scheduler.
2. Keep the zero-cost local engine as the default fallback.
3. Add a real local/self-hosted LLM only when a continuously online machine/server is available.
4. Add real Google Search Console analytics before using performance data for automatic learning.
5. Add stronger content-quality gates before increasing publishing volume.

