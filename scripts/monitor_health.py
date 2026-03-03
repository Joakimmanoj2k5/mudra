"""Health monitor loop for API with alert threshold."""

from __future__ import annotations

import argparse
import time
import urllib.request


def ping(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    args = parser.parse_args()

    failures = 0
    i = 0
    while True:
        ok = ping(f"{args.base}/health") and ping(f"{args.base}/ready")
        if ok:
            failures = 0
            print("HEALTHY")
        else:
            failures += 1
            print(f"UNHEALTHY consecutive_failures={failures}")
            if failures >= args.max_failures:
                print("ALERT: health threshold exceeded")
                return 1

        i += 1
        if args.iterations > 0 and i >= args.iterations:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
