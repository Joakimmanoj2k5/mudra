"""Lightweight API load test for /health and /ready endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import time
import urllib.request


def hit(url: str) -> tuple[bool, float, int]:
    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            code = int(resp.status)
            ok = code == 200
    except Exception:
        return False, (time.time() - start) * 1000.0, 0
    return ok, (time.time() - start) * 1000.0, code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    urls = [f"{args.base}/health" if i % 2 == 0 else f"{args.base}/ready" for i in range(args.requests)]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(hit, urls):
            results.append(r)

    oks = [r for r in results if r[0]]
    lat = [r[1] for r in results]
    success_rate = len(oks) / len(results) if results else 0.0
    p95 = sorted(lat)[int(0.95 * max(len(lat) - 1, 0))] if lat else 0.0

    print(f"requests={len(results)} success_rate={success_rate:.3f} avg_ms={(sum(lat)/len(lat)) if lat else 0:.2f} p95_ms={p95:.2f}")
    return 0 if success_rate >= 0.98 else 1


if __name__ == "__main__":
    raise SystemExit(main())
