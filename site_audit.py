import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

from engine.config import ConfigError, load_config
from engine.wp_client import get_posts, request


AUDIT_DIR = Path("data/audits")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def wc_html(html: str) -> int:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return len(re.findall(r"\b\w+\b", text))


def add_issue(issues: List[Dict[str, str]], severity: str, code: str, message: str, fix: str) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "recommended_fix": fix,
        }
    )


def safe_get(url: str, timeout: int = 20) -> requests.Response | None:
    try:
        return requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None


def public_site_checks(base_url: str, issues: List[Dict[str, str]], checks: Dict[str, Any]) -> None:
    home = safe_get(base_url, 25)
    checks["home_reachable"] = bool(home and home.status_code == 200)
    checks["home_status"] = home.status_code if home else 0
    if not home or home.status_code != 200:
        add_issue(
            issues,
            "critical",
            "site_unreachable",
            "Website homepage is not reachable (non-200 response).",
            "Fix domain/DNS/hosting uptime first. Without this, monetization is blocked.",
        )

    wp_json = safe_get(f"{base_url.rstrip('/')}/wp-json/", 25)
    checks["wp_json_reachable"] = bool(wp_json and wp_json.status_code == 200)
    if not checks["wp_json_reachable"]:
        add_issue(
            issues,
            "high",
            "wp_api_blocked",
            "WordPress REST API is not reachable.",
            "Enable REST API and remove security/firewall blocks for /wp-json/ endpoints.",
        )

    robots = safe_get(f"{base_url.rstrip('/')}/robots.txt", 20)
    checks["robots_status"] = robots.status_code if robots else 0
    if not robots or robots.status_code != 200:
        add_issue(
            issues,
            "medium",
            "robots_missing",
            "robots.txt is missing or inaccessible.",
            "Add robots.txt and allow important pages for crawl/index.",
        )

    sitemap = safe_get(f"{base_url.rstrip('/')}/wp-sitemap.xml", 20)
    checks["sitemap_status"] = sitemap.status_code if sitemap else 0
    if not sitemap or sitemap.status_code != 200:
        add_issue(
            issues,
            "high",
            "sitemap_missing",
            "wp-sitemap.xml is missing or inaccessible.",
            "Enable XML sitemap and submit it in Google Search Console.",
        )


def content_checks(
    base_url: str,
    wp_user: str,
    wp_app_password: str,
    timeout: int,
    issues: List[Dict[str, str]],
    checks: Dict[str, Any],
) -> None:
    posts = get_posts(base_url, wp_user, wp_app_password, timeout, per_page=50)
    checks["published_posts_count"] = len(posts)
    if len(posts) == 0:
        add_issue(
            issues,
            "critical",
            "no_published_posts",
            "No published posts found via WordPress API.",
            "Verify posting credentials, post status=publish, and scheduler execution.",
        )
        return

    title_counter = Counter()
    low_word = 0
    no_image = 0
    no_internal_links = 0
    missing_author_signature = 0
    categories_empty = 0
    tags_empty = 0

    for p in posts:
        title = re.sub(r"<[^>]+>", "", p.get("title", {}).get("rendered", "")).strip().lower()
        title_counter[title] += 1

        content_html = p.get("content", {}).get("rendered", "") or ""
        words = wc_html(content_html)
        if words < 350:
            low_word += 1

        if int(p.get("featured_media", 0) or 0) == 0:
            no_image += 1

        if content_html.count("<a ") < 2:
            no_internal_links += 1

        if "Author:</strong>" not in content_html or "About the Author</h3>" not in content_html:
            missing_author_signature += 1

        if not p.get("categories"):
            categories_empty += 1
        if not p.get("tags"):
            tags_empty += 1

    duplicate_titles = sum(c - 1 for c in title_counter.values() if c > 1)
    checks["duplicate_titles"] = duplicate_titles
    checks["low_word_posts"] = low_word
    checks["no_featured_image_posts"] = no_image
    checks["weak_internal_links_posts"] = no_internal_links
    checks["missing_author_signature_posts"] = missing_author_signature
    checks["empty_categories_posts"] = categories_empty
    checks["empty_tags_posts"] = tags_empty

    if duplicate_titles > 0:
        add_issue(
            issues,
            "high",
            "duplicate_titles",
            f"{duplicate_titles} duplicate post titles found.",
            "Enforce title uniqueness and topic rotation before publish.",
        )
    if low_word > 0:
        add_issue(
            issues,
            "medium",
            "thin_content",
            f"{low_word} posts appear thin (<350 words).",
            "Increase content depth, examples, and FAQs for ranking/ads quality.",
        )
    if no_image > 0:
        add_issue(
            issues,
            "medium",
            "missing_featured_images",
            f"{no_image} posts have no featured image.",
            "Attach featured images to improve CTR and ad engagement.",
        )
    if no_internal_links > 0:
        add_issue(
            issues,
            "medium",
            "weak_internal_linking",
            f"{no_internal_links} posts have weak internal linking (<2 links).",
            "Auto-insert at least 2 relevant internal links per post.",
        )
    if missing_author_signature > 0:
        add_issue(
            issues,
            "high",
            "author_signature_missing",
            f"{missing_author_signature} posts missing mandatory author blocks.",
            "Enforce author signature rule before publish and abort if missing.",
        )
    if categories_empty > 0 or tags_empty > 0:
        add_issue(
            issues,
            "medium",
            "taxonomy_incomplete",
            f"{categories_empty} posts missing categories and {tags_empty} missing tags.",
            "Auto-assign categories/tags by topic keywords for SEO structure.",
        )


def scheduler_checks(issues: List[Dict[str, str]], checks: Dict[str, Any]) -> None:
    path = Path("scheduler.log")
    checks["scheduler_log_exists"] = path.exists()
    if not path.exists():
        add_issue(
            issues,
            "high",
            "scheduler_log_missing",
            "scheduler.log not found.",
            "Run scheduler via GitHub Actions/worker and keep logs for monitoring.",
        )
        return

    tail = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-300:]
    fail_lines = [ln for ln in tail if "ok=False" in ln or "error" in ln.lower()]
    checks["recent_scheduler_errors"] = len(fail_lines)
    if len(fail_lines) > 0:
        add_issue(
            issues,
            "medium",
            "scheduler_failures",
            f"Recent scheduler failures/errors detected: {len(fail_lines)}",
            "Review scheduler.log failures and fix credentials/quota/topic duplication issues.",
        )


def compute_readiness(issues: List[Dict[str, str]]) -> Dict[str, Any]:
    weight = {"critical": 30, "high": 15, "medium": 7, "low": 3}
    penalty = sum(weight.get(i["severity"], 5) for i in issues)
    score = max(0, 100 - penalty)
    return {"monetization_readiness_score": score, "issue_count": len(issues)}


def save_report(report: Dict[str, Any]) -> str:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = AUDIT_DIR / name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2

    issues: List[Dict[str, str]] = []
    checks: Dict[str, Any] = {"audited_at": now_iso(), "wp_url": cfg.wp_url}

    public_site_checks(cfg.wp_url, issues, checks)
    try:
        content_checks(cfg.wp_url, cfg.wp_user, cfg.wp_app_password, cfg.request_timeout, issues, checks)
    except requests.RequestException:
        add_issue(
            issues,
            "critical",
            "wp_auth_failed",
            "Failed to fetch posts with current WordPress credentials.",
            "Re-check WP_USER/WP_APP_PASSWORD and app-password permissions.",
        )
    scheduler_checks(issues, checks)

    readiness = compute_readiness(issues)
    report = {"status": "ok", "checks": checks, "issues": issues, "summary": readiness}
    report_path = save_report(report)
    print(json.dumps({"status": "ok", "report_path": report_path, "summary": readiness}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
