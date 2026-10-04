# MRK Website Deployment

Continuous upgrades never deploy directly to production. They research, build, test, and record candidates.

Production deployment is a separate manual workflow:
- .github/workflows/mrk-deploy-pages.yml
- target: GitHub Pages
- trigger: workflow_dispatch
- authorization input: DEPLOY
- source_ref: explicit branch or tag
- GitHub Environment: github-pages

The workflow refuses an empty site or an unsafe deployment manifest. It does not use passwords, OTPs, payment details, or identity-verification bypasses.

Rollback is source-based: redeploy a previously approved branch/tag or commit. The manifest records the source commit and SHA-256 for every generated file.

The workflow is intentionally not scheduled.