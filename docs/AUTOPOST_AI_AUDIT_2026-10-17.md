# AutoPost AI repository audit — 2026-10-17

## Scope and evidence

Reviewed the public repository's default-branch tree, README and platform architecture, FastAPI bootstrap/configuration, authentication dependencies and routes, site model/creation route, automation service, dashboard UI, GitHub Actions validation workflow, and the most recent committed site-audit report (data/audits/audit_20261016_091118.json). This is a source review, not a live production penetration test. Runtime secrets, hosting environment, live deployment logs and actual GitHub Actions results were not accessible as part of this source review.

## Changes in this pull request

- Replaced the unrelated NovaMart storefront at repository root with a responsive AutoPost AI project landing page.
- Refreshed the FastAPI dashboard's existing CSS while retaining existing IDs, forms and JavaScript handlers to minimize functional regressions.
- Added basic regression tests for the two HTML entry points.

## Findings

### High — tenant-wide schedule/settings are stored in shared files

app/routes/ui.py allows any authenticated user to read/write config/schedule.json through /ui/schedule and /ui/apply-high-frequency-plan. The publishing and reporting handlers also operate on repository-global files and the global WordPress configuration. This is not tenant-isolated despite the multi-tenant architecture described in PLATFORM_ARCHITECTURE.md; one account may affect another account's schedule or shared automation state.

**Mitigation in this PR:** The legacy `/ui/*` control-panel API routes now require administrator authorization because they still read or mutate shared files/global configuration. Tenant-facing `/sites` and `/automation` routes continue to scope sites/jobs by the authenticated user. This is a temporary least-privilege barrier, not full tenant isolation for the legacy UI.

**Remaining fix:** Move schedule, SEO settings, job state and reporting into tenant/site-scoped database records. Scope every query and mutation to the authenticated user's site IDs, then selectively restore user-level access only for fully tenant-scoped handlers.

### High — credentials are stored without demonstrable encryption at rest

app/routes/sites.py assigns submitted WordPress Application Password and AI key strings directly to model fields named wp_app_password_enc and openai_api_key_enc. The automation service later calls decrypt_secret, but the create route itself does not encrypt the values before persistence. The _enc suffix therefore does not establish encryption.

**Recommended fix:** Encrypt secrets before database persistence using a deployment-managed key (for example, Fernet or a cloud KMS); never return secret fields in API responses; support key rotation and redact secrets from logs. Rotate credentials if real secrets were ever committed to source control.

### High — unsafe production security defaults

app/core/config.py defaults debug=True and jwt_secret_key='change-me'. A production deployment that omits environment configuration may therefore run with a predictable signing key. This could permit token forgery if that default is used in a reachable deployment.

**Recommended fix:** Fail startup in production when a strong secret is absent or equals a known placeholder; default debug to false; explicitly set environment-specific settings. Confirm the deployed environment and rotate the JWT secret if the placeholder has been used.

### Medium/High — administrator terminal endpoints execute shell commands

app/routes/ui.py exposes terminal execution routes that use subprocess.run(..., shell=True, ...). The general cloud terminal route is protected by require_admin, which is an important control, but a compromised admin account or token would provide command execution in the application environment.

**Recommended fix:** Disable these endpoints in production unless strictly required; otherwise isolate execution in a disposable sandbox, allowlist commands, restrict network/filesystem access, audit all invocations and enforce strict authorization.

### Medium — browser tokens are stored in localStorage

app/ui/index.html persists access and refresh tokens in browser localStorage. This is convenient but exposes bearer tokens to any successful same-origin script injection.

**Recommended fix:** Prefer secure, HttpOnly, SameSite cookies with CSRF protections where the deployment architecture supports them; otherwise minimize token lifetime, apply a strict Content Security Policy, avoid untrusted HTML insertion and provide token revocation.

### Medium — API/platform behavior lacks focused regression coverage

The current PR-validation workflow runs existing engine/upgrade tests and a generated-static-site smoke test, but there is no dedicated test suite visible for FastAPI authentication, tenant isolation, secret serialization, or the SaaS route layer.

**Recommended fix:** Add FastAPI TestClient tests for unauthenticated access, cross-tenant site/job access, schedule isolation, credential redaction, and production configuration validation. Add dependency/security scanning and Python compilation checks to CI.

### Operational — most recent committed site audit reports a critical website issue

The latest report records the configured WordPress homepage as unreachable, the sitemap inaccessible, missing author signatures on all three sampled posts, and a diagnostic SEO score of 33/100. The report's wp_url is https://dev-mrkofficial786.pantheonsite.io/. These are findings recorded in that report, not a fresh live check of the site.

**Recommended fix:** Re-run the site audit against the intended production domain, resolve DNS/hosting and sitemap availability first, then enforce author-signature and metadata quality gates before increasing publishing volume.

## Deployment and verification notes

- The repository root page and FastAPI-served dashboard are separate entry points: root index.html is a static landing page, while / in the FastAPI app serves app/ui/index.html when that backend is deployed.
- The landing-page metrics/chart are explicitly illustrative, not live analytics.
- This PR does not merge to main, change production credentials, enable deployment, or claim a successful production rollout.
- After CI passes and the changes are reviewed, verify the actual Vercel/static hosting target and the FastAPI backend separately. A static landing page deployment does not itself deploy the API or its authenticated dashboard.
