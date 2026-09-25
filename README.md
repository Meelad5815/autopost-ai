# MRK Autopost AI

Zero-cost remote WordPress autoposting system designed to run from **GitHub Actions**, so the user's personal computer does not need to stay powered on.

## How it runs

GitHub Actions starts short-lived Ubuntu runners at the configured UTC times:

- 10:00 PKT
- 12:00 PKT
- 14:00 PKT
- 16:00 PKT
- 18:00 PKT

Each slot runs the scheduler once. The scheduler can create one post in each configured language, then performs the update loop. The personal Windows PC is not required for scheduled production runs.

## AI modes

### Default: local zero-cost engine

`AI_PROVIDER=local`

This mode requires no OpenAI, Gemini, or other paid AI API. It uses the repository's deterministic local content-intelligence engine and can run inside GitHub Actions.

Important: this is not a neural LLM equivalent to ChatGPT. It is a deterministic generation system designed for zero-cost automation.

### Optional: self-hosted Ollama

`AI_PROVIDER=ollama`

Ollama can provide a real local open-source LLM on a computer/server that stays online. A personal PC that is switched off cannot provide Ollama inference to GitHub Actions.

## Required GitHub Secrets

Add these repository secrets for WordPress publishing:

- `WP_URL`
- `WP_USER`
- `WP_APP_PASSWORD`

No paid AI API key is required for the GitHub Actions workflow.

## Google Search Console learning loop

The repository now includes an optional Google Search Console collector:

- `engine/search_console.py`
- `.github/workflows/search-console.yml`
- output: `data/search_console.json`

It uses the official Search Console Search Analytics API to collect real query/page clicks, impressions, CTR and position data for a configurable recent window. Search Console requires authorization with the `webmasters.readonly` scope. The API may return only the top rows available under its data limits, so the stored dataset is not treated as a complete export.

Configure these GitHub repository secrets before enabling the workflow:

- `GSC_SITE_URL` — exactly the property URL or `sc-domain:example.com` value used by Search Console.
- `GSC_SERVICE_ACCOUNT_JSON` — the Google Cloud service-account JSON credential.

The service-account email must have access to the Search Console property. Do not put credentials in source files or chat.

The collector intentionally does not fabricate analytics. If credentials are missing, it fails the dedicated workflow instead of inventing traffic numbers.

## Publishing and social queue

After a WordPress post is created, the system prepares platform-specific social variants and writes them to `data/social_queue.json`.

The current queue supports:

- Facebook
- Instagram
- X
- LinkedIn
- Telegram
- Pinterest
- YouTube

The queue does not claim to publish through personal WhatsApp, WhatsApp Status, or arbitrary WhatsApp groups. Official platform authorization/API integration is required before direct publishing.

## SEO and quality controls

The system includes:

- technical SEO signal checks
- title/meta/canonical/H1/image-alt checks
- explainable SEO health scoring
- Article JSON-LD generation
- content quality gates
- Google research feeds
- local learning signals
- real Search Console analytics collection

These scores are diagnostic signals; they are not guarantees of Google rankings or AdSense approval.

## Important operational notes

- GitHub Actions is a scheduled job runner, not a permanently running process.
- Scheduled runs can occasionally be delayed by GitHub.
- WordPress must be reachable from the internet.
- WordPress Application Password credentials must remain in GitHub Secrets, never in source code.
- Search Console credentials must remain in GitHub Secrets.
- The workflow stores runtime history/trends/keywords back into the repository.
- Traffic metrics are never intentionally fabricated.

## Recommended production path

1. Keep GitHub Actions as the remote scheduler.
2. Keep the zero-cost local engine as the default fallback.
3. Keep real Search Console analytics as the source of performance feedback.
4. Add stronger content-quality gates before increasing publishing volume.
5. Add direct social API publishing only after each platform's official authorization is configured.
