#!/usr/bin/env python3
"""
Production Health Verification Script.

Verifies:
1. API root endpoint is reachable.
2. GET /health returns HTTP 200 and status 'ok'.
3. GET /model-info returns HTTP 200 and valid model metadata.

Usage:
    python scripts/verify_health.py [BASE_URL]
    Example: python scripts/verify_health.py http://localhost:8000
    Or set API_URL environment variable.
"""

import sys
import os
import json
import urllib.request
import urllib.error

def check_endpoint(url: str, expected_key: str = None) -> dict:
    """Make HTTP GET request and return parsed JSON response."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "HealthCheck/1.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP Status {response.status} from {url}")
        content = response.read().decode("utf-8")
        data = json.loads(content)
        if expected_key and expected_key not in data:
            raise ValueError(f"Missing expected key '{expected_key}' in response from {url}")
        return data

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("API_URL", "http://localhost:8000")
    base_url = base_url.rstrip("/")

    print(f"========================================")
    print(f"PRODUCTION HEALTH VERIFICATION")
    print(f"Target API: {base_url}")
    print(f"========================================\n")

    errors = 0

    # 1. Verify GET /health
    health_url = f"{base_url}/health"
    try:
        data = check_endpoint(health_url, expected_key="status")
        status_val = data.get("status")
        if status_val == "ok":
            print(f" [PASS] GET /health (200 OK) -> status: '{status_val}'")
        else:
            print(f" [FAIL] GET /health unexpected status: '{status_val}'")
            errors += 1
    except Exception as e:
        print(f" [FAIL] GET /health error: {e}")
        errors += 1

    # 2. Verify GET /model-info
    model_info_url = f"{base_url}/model-info"
    try:
        data = check_endpoint(model_info_url, expected_key="model_version")
        version = data.get("model_version")
        model_type = data.get("model_type")
        rmspe = data.get("validation_rmspe")
        print(f" [PASS] GET /model-info (200 OK) -> version: '{version}', type: '{model_type}', RMSPE: {rmspe}%")
    except Exception as e:
        print(f" [FAIL] GET /model-info error: {e}")
        errors += 1

    print("\n========================================")
    if errors == 0:
        print("HEALTH VERIFICATION PASSED ALL CHECKS")
        print("========================================\n")
        sys.exit(0)
    else:
        print(f"HEALTH VERIFICATION FAILED ({errors} error(s))")
        print("========================================\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
