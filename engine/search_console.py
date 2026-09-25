import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import requests

OUTPUT_FILE = Path("data/search_console.json")
TOKEN_URL = "https://oauth2.googleapis.com/token"
API_BASE = "https://www.googleapis.com/webmasters/v3/sites"


def _credentials() -> Dict[str, Any]:
    raw = os.getenv("GSC_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise RuntimeError("GSC_SERVICE_ACCOUNT_JSON is not configured.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GSC_SERVICE_ACCOUNT_JSON is not valid JSON.") from exc


def _access_token(credentials: Dict[str, Any]) -> str:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(
        credentials,
        scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
    )
    creds.refresh(Request())
    if not creds.token:
        raise RuntimeError("Google did not return an access token.")
    return creds.token


def query_search_analytics(
    site_url: str,
    access_token: str,
    start_date: str,
    end_date: str,
    dimensions: List[str],
    row_limit: int = 25000,
) -> List[Dict[str, Any]]:
    from urllib.parse import quote

    encoded_site = quote(site_url, safe="")
    url = f"{API_BASE}/{encoded_site}/searchAnalytics/query"
    payload = {
        "startDate": start_date,
        "endDate": end_date,
        "dimensions": dimensions,
        "rowLimit": row_limit,
        "startRow": 0,
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("rows", [])


def collect(days: int = 28) -> Dict[str, Any]:
    site_url = os.getenv("GSC_SITE_URL", "").strip()
    if not site_url:
        raise RuntimeError("GSC_SITE_URL is not configured.")

    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=max(1, days) - 1)
    token = _access_token(_credentials())

    query_rows = query_search_analytics(
        site_url, token, start.isoformat(), end.isoformat(), ["query"], 25000
    )
    page_rows = query_search_analytics(
        site_url, token, start.isoformat(), end.isoformat(), ["page"], 25000
    )

    return {
        "site_url": site_url,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "source": "Google Search Console Search Analytics API",
        "query_rows": query_rows,
        "page_rows": page_rows,
        "totals": {
            "query_clicks": round(sum(float(r.get("clicks", 0)) for r in query_rows), 2),
            "query_impressions": round(sum(float(r.get("impressions", 0)) for r in query_rows), 2),
            "page_clicks": round(sum(float(r.get("clicks", 0)) for r in page_rows), 2),
            "page_impressions": round(sum(float(r.get("impressions", 0)) for r in page_rows), 2),
        },
    }


def save(payload: Dict[str, Any]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    days = int(os.getenv("GSC_DAYS", "28"))
    save(collect(days))
    print(f"Saved Google Search Console data to {OUTPUT_FILE}")
