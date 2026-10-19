"""Run an explicit Bright Data connectivity check.

This is a manual diagnostic, not part of the automated test suite.
"""

import json
import os
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    api_token = os.environ.get("BRIGHTDATA_API_KEY")
    if not api_token:
        print("BRIGHTDATA_API_KEY was not found in .env")
        return 1

    query = "site:youtube.com funny cat videos"
    params = urlencode({"q": query, "hl": "en", "gl": "us"})
    target_url = f"https://www.google.com/search?{params}&brd_json=1"
    response = requests.post(
        "https://api.brightdata.com/request",
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        json={"zone": "serp_api1", "url": target_url, "format": "json"},
        timeout=60,
    )

    print(f"Status: {response.status_code}")
    if not response.ok:
        print(response.text[:500])
        return 1

    data = response.json()
    body = data.get("body", data)
    if isinstance(body, str):
        body = json.loads(body)

    organic = body.get("organic", []) if isinstance(body, dict) else []
    print(f"Organic results: {len(organic)}")
    for result in organic[:3]:
        print(f"- {result.get('title')}: {result.get('link') or result.get('url')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
